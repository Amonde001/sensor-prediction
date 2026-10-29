"""Reusable loader for the soil-moisture sensor workbook."""

import pandas as pd


class SoilMoistureData:
    """Load and normalize the unprocessed soil-moisture sensor sheet."""

    COLUMN_NAMES = ["valid_time", "Port 1", "Port 2", "Port 3", "Port 4", "Port 5"]

    def __init__(self, file_path):
        self.file_path = file_path
        self.df = self._load_data()

    def _load_data(self):
        """Read the workbook, parse timestamps, and coerce sensor values to numbers."""
        dataframe = pd.read_excel(
            self.file_path,
            skiprows=3,
            names=self.COLUMN_NAMES,
            sheet_name="Unprocessed Data",
        )
        dataframe["valid_time"] = pd.to_datetime(dataframe["valid_time"])
        sensor_columns = dataframe.columns[1:]
        dataframe[sensor_columns] = dataframe[sensor_columns].apply(pd.to_numeric, errors="coerce")
        return dataframe

    def get_data(self):
        """Return the normalized sensor dataframe."""
        return self.df