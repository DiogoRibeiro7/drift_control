from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import pandas as pd
from dataexcept import DataLoadingError, FileWriteError

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9._-]+$")
_FORMAT_SUFFIXES: tuple[Literal["parquet", "csv"], ...] = ("parquet", "csv")


def _validate_component(value: str, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    if not _SAFE_COMPONENT.match(value):
        raise ValueError(
            f"{field} may only contain letters, digits, '.', '_' and '-'; got {value!r}"
        )


def _file_extension(fmt: Literal["parquet", "csv"]) -> str:
    return "parquet" if fmt == "parquet" else "csv"


def _baseline_filename(name: str, version: str, fmt: Literal["parquet", "csv"]) -> str:
    _validate_component(name, "name")
    _validate_component(version, "version")
    return f"{name}_v{version}.{_file_extension(fmt)}"


def _metadata_payload(
    *,
    name: str,
    version: str,
    path: str,
    fmt: Literal["parquet", "csv"],
    row_count: int,
    dataset_sha256: str,
    owner: str | None,
    training_job_id: str | None,
) -> dict[str, Any]:
    return {
        "name": name,
        "version": version,
        "path": path,
        "format": fmt,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": row_count,
        "dataset_sha256": dataset_sha256,
        "owner": owner,
        "training_job_id": training_job_id,
    }


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


def _select_format(
    requested: Literal["parquet", "csv"] | None,
    default_format: Literal["parquet", "csv"],
) -> Literal["parquet", "csv"]:
    selected = requested or default_format
    if selected == "parquet" and not _parquet_available():
        return "csv"
    return selected


def _bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_frame_from_bytes(payload: bytes, fmt: Literal["parquet", "csv"]) -> pd.DataFrame:
    buffer = BytesIO(payload)
    if fmt == "parquet":
        return pd.read_parquet(buffer)
    return pd.read_csv(buffer)


def _write_frame_to_bytes(data: pd.DataFrame, fmt: Literal["parquet", "csv"]) -> bytes:
    buffer = BytesIO()
    if fmt == "parquet":
        data.to_parquet(buffer, index=False)
    else:
        buffer.write(data.to_csv(index=False).encode("utf-8"))
    return buffer.getvalue()


def _parse_baseline_identifier(stem: str) -> tuple[str, str] | None:
    if "_v" not in stem:
        return None
    name, version = stem.rsplit("_v", 1)
    return name, version


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
        return os.path.join(self.directory, _baseline_filename(name, version, fmt))

    def _meta_path(self, name: str, version: str, fmt: Literal["parquet", "csv"] = "csv") -> str:
        return self._path(name, version, fmt=fmt) + ".meta.json"

    @staticmethod
    def _file_sha256(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _resolve_existing_path(
        self, name: str, version: str
    ) -> tuple[str, Literal["parquet", "csv"]]:
        for fmt in _FORMAT_SUFFIXES:
            path = self._path(name, version, fmt=fmt)
            if os.path.exists(path):
                return path, fmt
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
        selected = _select_format(fmt, self.default_format)
        path = self._path(name, version, fmt=selected)
        try:
            if selected == "parquet":
                data.to_parquet(path, index=False)
            else:
                data.to_csv(path, index=False)
        except OSError as exc:
            raise FileWriteError(path, original=exc) from exc
        try:
            digest = self._file_sha256(path)
        except OSError as exc:
            raise DataLoadingError(path, exc) from exc
        meta = _metadata_payload(
            name=name,
            version=version,
            path=path,
            fmt=selected,
            row_count=int(len(data)),
            dataset_sha256=digest,
            owner=owner,
            training_job_id=training_job_id,
        )
        meta_path = self._meta_path(name, version, fmt=selected)
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f)
        except OSError as exc:
            raise FileWriteError(meta_path, original=exc) from exc
        return path

    def load(self, name: str, version: str, verify_integrity: bool = True) -> pd.DataFrame:
        path, resolved_fmt = self._resolve_existing_path(name, version)
        if verify_integrity:
            meta = self.metadata(name, version)
            expected = str(meta.get("dataset_sha256", ""))
            try:
                actual = self._file_sha256(path)
            except OSError as exc:
                raise DataLoadingError(path, exc) from exc
            if expected and expected != actual:
                raise ValueError(
                    f"Integrity check failed for baseline {name} v{version}: "
                    f"expected {expected}, got {actual}"
                )
        try:
            if resolved_fmt == "parquet":
                return pd.read_parquet(path)
            return pd.read_csv(path)
        except (OSError, ValueError) as exc:
            # Pandas uses ValueError for empty or malformed tabular input.
            raise DataLoadingError(path, exc) from exc

    def list(self, name: str | None = None) -> list[str]:
        out: list[str] = []
        root = Path(self.directory)
        for fmt in _FORMAT_SUFFIXES:
            for path in root.glob(f"*_v*.{_file_extension(fmt)}"):
                identifier = _parse_baseline_identifier(path.stem)
                if identifier is None:
                    continue
                baseline_name, version = identifier
                if name is not None and baseline_name != name:
                    continue
                out.append(f"{baseline_name}@{version}")
        return sorted(out)

    def metadata(self, name: str, version: str) -> dict:
        for fmt in _FORMAT_SUFFIXES:
            meta_path = self._meta_path(name, version, fmt=fmt)
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, encoding="utf-8") as f:
                        return cast("dict[Any, Any]", json.load(f))
                except (OSError, ValueError) as exc:
                    raise DataLoadingError(meta_path, exc) from exc
        raise FileNotFoundError(f"Metadata for baseline {name} v{version} not found")

    def delete(self, name: str, version: str) -> None:
        for fmt in _FORMAT_SUFFIXES:
            for path in (
                self._path(name, version, fmt=fmt),
                self._meta_path(name, version, fmt=fmt),
            ):
                if os.path.exists(path):
                    os.remove(path)


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
        filename = _baseline_filename(name, version, fmt)
        return f"{self.prefix}/{filename}" if self.prefix else filename

    def _meta_key(self, name: str, version: str, fmt: Literal["parquet", "csv"] = "csv") -> str:
        return self._object_key(name, version, fmt=fmt) + ".meta.json"

    def _read_object_bytes(self, key: str) -> bytes:
        obj = self.s3_client.get_object(Bucket=self.bucket, Key=key)
        return cast(bytes, obj["Body"].read())

    def _exists(self, key: str) -> bool:
        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False

    def _resolve_existing_key(
        self, name: str, version: str
    ) -> tuple[str, Literal["parquet", "csv"]]:
        for fmt in _FORMAT_SUFFIXES:
            key = self._object_key(name, version, fmt=fmt)
            if self._exists(key):
                return key, fmt
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
        selected = _select_format(fmt, self.default_format)
        key = self._object_key(name, version, fmt=selected)
        payload = _write_frame_to_bytes(data, selected)
        self.s3_client.put_object(Bucket=self.bucket, Key=key, Body=payload)
        path = f"s3://{self.bucket}/{key}"
        meta = _metadata_payload(
            name=name,
            version=version,
            path=path,
            fmt=selected,
            row_count=int(len(data)),
            dataset_sha256=_bytes_sha256(payload),
            owner=owner,
            training_job_id=training_job_id,
        )
        self.s3_client.put_object(
            Bucket=self.bucket,
            Key=self._meta_key(name, version, fmt=selected),
            Body=json.dumps(meta).encode("utf-8"),
        )
        return path

    def load(self, name: str, version: str, verify_integrity: bool = True) -> pd.DataFrame:
        key, resolved_fmt = self._resolve_existing_key(name, version)
        payload = self._read_object_bytes(key)
        if verify_integrity:
            meta = self.metadata(name, version)
            expected = str(meta.get("dataset_sha256", ""))
            actual = _bytes_sha256(payload)
            if expected and expected != actual:
                raise ValueError(
                    f"Integrity check failed for baseline {name} v{version}: "
                    f"expected {expected}, got {actual}"
                )
        return _read_frame_from_bytes(payload, resolved_fmt)

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
                for fmt in _FORMAT_SUFFIXES:
                    suffix = f".{_file_extension(fmt)}"
                    if stem.endswith(suffix):
                        stem = stem[: -len(suffix)]
                        break
                else:
                    continue
                identifier = _parse_baseline_identifier(stem)
                if identifier is None:
                    continue
                baseline_name, version = identifier
                if name is not None and baseline_name != name:
                    continue
                out.append(f"{baseline_name}@{version}")
        return sorted(out)

    def metadata(self, name: str, version: str) -> dict:
        for fmt in _FORMAT_SUFFIXES:
            key = self._meta_key(name, version, fmt=fmt)
            if self._exists(key):
                return cast(
                    "dict[Any, Any]",
                    json.loads(self._read_object_bytes(key).decode("utf-8")),
                )
        raise FileNotFoundError(f"Metadata for baseline {name} v{version} not found")

    def delete(self, name: str, version: str) -> None:
        for fmt in _FORMAT_SUFFIXES:
            for key in (
                self._object_key(name, version, fmt=fmt),
                self._meta_key(name, version, fmt=fmt),
            ):
                if self._exists(key):
                    self.s3_client.delete_object(Bucket=self.bucket, Key=key)
