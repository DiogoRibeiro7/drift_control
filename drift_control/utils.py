import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads data from a CSV file.

    :param file_path: The path to the CSV file.
    :return: DataFrame containing the data.
    """
    return pd.read_csv(file_path)

def save_data(data: pd.DataFrame, file_path: str) -> None:
    """
    Saves data to a CSV file.

    :param data: DataFrame containing the data to save.
    :param file_path: The path to the CSV file.
    """
    data.to_csv(file_path, index=False)
