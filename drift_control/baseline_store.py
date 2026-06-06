import hashlib
import json
import os
import re
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Literal, Protocol

import pandas as pd

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9._-]+$")


def _validate_component(value: str, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    if not _SAFE_COMPONENT.match(value):
        raise ValueError(
            f"{field} may only contain letters, digits, '.', '_' and '-'; "
            f"got {value!r}"
        )


class BaselineStore(Protocol):
    def save(
        self,
        data: pd.DataFrame,
        name: str,
        version: str,
        owner: str | None = None,
        training_job_id: str | None = None,
        fmt: Literal["parquet", "csv"] | None = None,
    ) -> str: ...

    def load(self, name: str, version: str, verify_integrity: bool = True) -> pd.DataFrame: ...

    def list(self, name: str | None = None) -> list[str]: ...

    def metadata(self, name: str, version: str) -> dict: ...

    def delete(self, name: str, version: str) -> None: ...


class LocalBaselineStore:
    def __init__(
        self,
        directory: str = "baselines",
        default_format: Literal["parquet", "csv"] = "parquet",
    ) -> None:
        self.directory = directory
        self.default_format = default_format
        os.makedirs(self.directory, exist_ok=True)

    def _path(self, name: str, version: str, fmt: Literal["parquet", "csv"] = "csv") -> str:
        _validate_component(name, "name")
        _validate_component(version, "version")
        ext = "parquet" if fmt == "parquet" else "csv"
        filename = f"{name}_v{version}.{ext}"
        return os.path.join(self.directory, filename)

    @staticmethod
    def _file_sha256(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _meta_path(self, name: str, version: str, fmt: Literal["parquet", "csv"] = "csv") -> str:
        return self._path(name, version, fmt=fmt) + ".meta.json"

    @staticmethod
    def _parquet_available() -> bool:
        try:
            import pyarrow  # type: ignore # noqa: F401

            return True
        except Exception:
            try:
                import fastparquet  # type: ignore # noqa: F401

                return True
            except Exception:
                return False

    def _resolve_existing_path(self, name: str, version: str) -> tuple[str, Literal["parquet", "csv"]]:
        p_parquet = self._path(name, version, fmt="parquet")
        p_csv = self._path(name, version, fmt="csv")
        if os.path.exists(p_parquet):
            return p_parquet, "parquet"
        if os.path.exists(p_csv):
            return p_csv, "csv"
        raise FileNotFoundError(f"Baseline {name} v{version} not found")

    def save(
        self,
        data: pd.DataFrame,
        name: str,
        version: str,
        owner: str | None = None,
        training_job_id: str | None = None,
        fmt: Literal["parquet", "csv"] | None = None,
    ) -> str:
        selected = fmt or self.default_format
        if selected == "parquet" and not self._parquet_available():
            selected = "csv"
        path = self._path(name, version, fmt=selected)
        if selected == "parquet":
            data.to_parquet(path, index=False)
        else:
            data.to_csv(path, index=False)
        meta = {
            "name": name,
            "version": version,
            "path": path,
            "format": selected,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "row_count": int(len(data)),
            "dataset_sha256": self._file_sha256(path),
            "owner": owner,
            "training_job_id": training_job_id,
        }
        with open(self._meta_path(name, version, fmt=selected), "w", encoding="utf-8") as f:
            json.dump(meta, f)
        return path

    def load(self, name: str, version: str, verify_integrity: bool = True) -> pd.DataFrame:
        path, resolved_fmt = self._resolve_existing_path(name, version)
        if verify_integrity:
            meta = self.metadata(name, version)
            expected = str(meta.get("dataset_sha256", ""))
            actual = self._file_sha256(path)
            if expected and expected != actual:
                raise ValueError(
                    f"Integrity check failed for baseline {name} v{version}: "
                    f"expected {expected}, got {actual}"
                )
        if resolved_fmt == "parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)

    def list(self, name: str | None = None) -> list[str]:
        out: list[str] = []
        for p in list(Path(self.directory).glob("*_v*.csv")) + list(Path(self.directory).glob("*_v*.parquet")):
            stem = p.stem
            if "_v" not in stem:
                continue
            n, v = stem.rsplit("_v", 1)
            if name is not None and n != name:
                continue
            out.append(f"{n}@{v}")
        return sorted(out)

    def metadata(self, name: str, version: str) -> dict:
        meta_path_csv = self._meta_path(name, version, fmt="csv")
        meta_path_parquet = self._meta_path(name, version, fmt="parquet")
        meta_path = meta_path_parquet if os.path.exists(meta_path_parquet) else meta_path_csv
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata for baseline {name} v{version} not found")
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)

    def delete(self, name: str, version: str) -> None:
        for fmt in ("csv", "parquet"):
            path = self._path(name, version, fmt=fmt)
            meta_path = self._meta_path(name, version, fmt=fmt)
            if os.path.exists(path):
                os.remove(path)
            if os.path.exists(meta_path):
                os.remove(meta_path)


class S3BaselineStore:
    def __init__(
        self,
        bucket: str,
        prefix: str = "baselines",
        default_format: Literal["parquet", "csv"] = "parquet",
        s3_client: object | None = None,
    ) -> None:
        if not bucket:
            raise ValueError("bucket must be a non-empty string")
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.default_format = default_format
        if s3_client is None:
            try:
                import boto3  # type: ignore
            except Exception as exc:
                raise RuntimeError("boto3 is required for S3BaselineStore") from exc
            self.s3_client = boto3.client("s3")
        else:
            self.s3_client = s3_client

    def _object_key(self, name: str, version: str, fmt: Literal["parquet", "csv"] = "csv") -> str:
        _validate_component(name, "name")
        _validate_component(version, "version")
        ext = "parquet" if fmt == "parquet" else "csv"
        base = f"{name}_v{version}.{ext}"
        return f"{self.prefix}/{base}" if self.prefix else base

    def _meta_key(self, name: str, version: str, fmt: Literal["parquet", "csv"] = "csv") -> str:
        return self._object_key(name, version, fmt=fmt) + ".meta.json"

    @staticmethod
    def _parquet_available() -> bool:
        try:
            import pyarrow  # type: ignore # noqa: F401
            return True
        except Exception:
            try:
                import fastparquet  # type: ignore # noqa: F401
                return True
            except Exception:
                return False

    @staticmethod
    def _bytes_sha256(payload: bytes) -> str:
        return hashlib.sha256(payload).hexdigest()

    def _read_object_bytes(self, key: str) -> bytes:
        obj = self.s3_client.get_object(Bucket=self.bucket, Key=key)
        return obj["Body"].read()

    def _exists(self, key: str) -> bool:
        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False

    def _resolve_existing_key(self, name: str, version: str) -> tuple[str, Literal["parquet", "csv"]]:
        k_parquet = self._object_key(name, version, fmt="parquet")
        k_csv = self._object_key(name, version, fmt="csv")
        if self._exists(k_parquet):
            return k_parquet, "parquet"
        if self._exists(k_csv):
            return k_csv, "csv"
        raise FileNotFoundError(f"Baseline {name} v{version} not found")

    def save(
        self,
        data: pd.DataFrame,
        name: str,
        version: str,
        owner: str | None = None,
        training_job_id: str | None = None,
        fmt: Literal["parquet", "csv"] | None = None,
    ) -> str:
        selected = fmt or self.default_format
        if selected == "parquet" and not self._parquet_available():
            selected = "csv"
        key = self._object_key(name, version, fmt=selected)
        buf = BytesIO()
        if selected == "parquet":
            data.to_parquet(buf, index=False)
        else:
            csv_text = data.to_csv(index=False)
            buf.write(csv_text.encode("utf-8"))
        payload = buf.getvalue()
        self.s3_client.put_object(Bucket=self.bucket, Key=key, Body=payload)
        meta = {
            "name": name,
            "version": version,
            "path": f"s3://{self.bucket}/{key}",
            "format": selected,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "row_count": int(len(data)),
            "dataset_sha256": self._bytes_sha256(payload),
            "owner": owner,
            "training_job_id": training_job_id,
        }
        self.s3_client.put_object(
            Bucket=self.bucket,
            Key=self._meta_key(name, version, fmt=selected),
            Body=json.dumps(meta).encode("utf-8"),
        )
        return meta["path"]

    def load(self, name: str, version: str, verify_integrity: bool = True) -> pd.DataFrame:
        key, resolved_fmt = self._resolve_existing_key(name, version)
        payload = self._read_object_bytes(key)
        if verify_integrity:
            meta = self.metadata(name, version)
            expected = str(meta.get("dataset_sha256", ""))
            actual = self._bytes_sha256(payload)
            if expected and expected != actual:
                raise ValueError(
                    f"Integrity check failed for baseline {name} v{version}: "
                    f"expected {expected}, got {actual}"
                )
        buf = BytesIO(payload)
        if resolved_fmt == "parquet":
            return pd.read_parquet(buf)
        return pd.read_csv(buf)

    def list(self, name: str | None = None) -> list[str]:
        prefix = f"{self.prefix}/" if self.prefix else ""
        paginator = self.s3_client.get_paginator("list_objects_v2")
        out: list[str] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                key = item.get("Key", "")
                if key.endswith(".meta.json"):
                    continue
                stem = key.split("/")[-1]
                if stem.endswith(".csv"):
                    stem = stem[:-4]
                elif stem.endswith(".parquet"):
                    stem = stem[:-8]
                else:
                    continue
                if "_v" not in stem:
                    continue
                n, v = stem.rsplit("_v", 1)
                if name is not None and n != name:
                    continue
                out.append(f"{n}@{v}")
        return sorted(out)

    def metadata(self, name: str, version: str) -> dict:
        mk_parquet = self._meta_key(name, version, fmt="parquet")
        mk_csv = self._meta_key(name, version, fmt="csv")
        key = mk_parquet if self._exists(mk_parquet) else mk_csv
        if not self._exists(key):
            raise FileNotFoundError(f"Metadata for baseline {name} v{version} not found")
        return json.loads(self._read_object_bytes(key).decode("utf-8"))

    def delete(self, name: str, version: str) -> None:
        for fmt in ("csv", "parquet"):
            for key in (self._object_key(name, version, fmt=fmt), self._meta_key(name, version, fmt=fmt)):
                if self._exists(key):
                    self.s3_client.delete_object(Bucket=self.bucket, Key=key)

