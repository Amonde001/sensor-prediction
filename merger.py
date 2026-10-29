#merger.py used to merge the climate and moisture data. It is a reusable class for merging two or more pandas DataFrames, with features like argument validation, progress bar, and colored terminal messages.

import time

import pandas as pd
from colorama import Fore, init


# Initialize colorama
init(autoreset=True)


class merger:
    """
    Merge two or more pandas DataFrames on a shared column.

    Parameters
    ----------
    dataframes : list, tuple, or dict
        Two or more pandas DataFrames. A dictionary may be used to give
        each DataFrame a display name.

    merge_column : str
        Column used to merge all DataFrames.

    how : str, default="inner"
        Pandas merge method. Supported values are inner, left, right,
        and outer.

    suffixes : tuple, default=("", "_duplicate")
        Suffixes applied to overlapping non-key columns.
    """

    SUPPORTED_MERGE_TYPES = {"inner", "left", "right", "outer"}

    def __init__(
        self,
        dataframes,
        merge_column,
        how="inner",
        suffixes=("", "_duplicate")
    ):
        """Initialize the merger and run the complete merge workflow."""
        self.dataframes = dataframes
        self.merge_column = (
            merge_column.strip()
            if isinstance(merge_column, str)
            else merge_column
        )
        self.how = how.lower().strip() if isinstance(how, str) else how
        self.suffixes = suffixes

        self.dataframe_items = []
        self.combined_data = None
        self.errors = []
        self.warnings = []

        self._run_full_system()

    # ==========================================================
    # GENERAL DISPLAY METHODS
    # ==========================================================

    def _pause(self, seconds=0.2):
        """Pause execution for the specified number of seconds."""
        time.sleep(seconds)

    def _header(self, title):
        """Display a major section header."""
        print()
        print(Fore.CYAN + "-" * 70)
        print(Fore.CYAN + title.center(70))
        print(Fore.CYAN + "-" * 70)
        self._pause()

    def _subheader(self, title):
        """Display a subsection header."""
        print()
        print(Fore.YELLOW + "-" * 70)
        print(Fore.YELLOW + title)
        print(Fore.YELLOW + "-" * 70)
        self._pause()

    def _info(self, message):
        """Display an informational message."""
        print(Fore.WHITE + f"[INFO] {message}")
        self._pause()

    def _success(self, message):
        """Display a successful operation message."""
        print(Fore.GREEN + f"[SUCCESS] {message}")
        self._pause()

    def _warning(self, message):
        """Display a warning and retain it in the warning list."""
        self.warnings.append(str(message))
        print(Fore.YELLOW + f"[WARNING] {message}")
        self._pause()

    def _error(self, message):
        """Display an error and retain it in the error list."""
        self.errors.append(str(message))
        print(Fore.RED + f"[ERROR] {message}")
        self._pause()

    # ==========================================================
    # COMPLETE WORKFLOW
    # ==========================================================

    def _run_full_system(self):
        """Execute validation, merging, and final reporting."""
        self._header("DATA MERGING SYSTEM")
        self._info("Starting data merging process...")

        if not self._validate_arguments():
            self._error("Validation failed. Data merging stopped.")
            self._final_report()
            return

        if self.merge_data() is None:
            self._warning("Data merging did not produce a combined DataFrame.")

        self._final_report()

    # ==========================================================
    # STEP 1: VALIDATION
    # ==========================================================

    def _validate_arguments(self):
        """Validate all merger arguments and normalize DataFrame names."""
        self._header("VALIDATING INPUTS")
        valid = True

        self._info(f"Checking merge column: {self.merge_column}")
        if not isinstance(self.merge_column, str) or not self.merge_column.strip():
            self._error("merge_column must be a non-empty string.")
            valid = False
        else:
            self._success(f"Merge column '{self.merge_column}' is valid.")

        self._info(f"Checking merge type: {self.how}")
        if self.how not in self.SUPPORTED_MERGE_TYPES:
            self._error(
                "Unsupported merge type. Use 'inner', 'left', 'right', or 'outer'."
            )
            valid = False
        else:
            self._success(f"Merge type '{self.how}' is supported.")

        self._info("Checking suffixes...")
        if (
            not isinstance(self.suffixes, tuple)
            or len(self.suffixes) != 2
            or any(not isinstance(suffix, str) for suffix in self.suffixes)
        ):
            self._error("suffixes must be a tuple containing two strings.")
            valid = False
        else:
            self._success("Suffixes are valid.")

        self._info("Checking DataFrames...")
        if isinstance(self.dataframes, dict):
            dataframe_items = list(self.dataframes.items())
        elif isinstance(self.dataframes, (list, tuple)):
            dataframe_items = [
                (f"DataFrame {number}", dataframe)
                for number, dataframe in enumerate(self.dataframes, start=1)
            ]
        else:
            self._error("dataframes must be a list, tuple, or dictionary.")
            dataframe_items = []
            valid = False

        if len(dataframe_items) < 2:
            self._error("At least two DataFrames are required for merging.")
            valid = False

        for name, dataframe in dataframe_items:
            if not isinstance(dataframe, pd.DataFrame):
                self._error(f"{name} is not a pandas DataFrame.")
                valid = False
                continue

            if self.merge_column not in dataframe.columns:
                self._error(
                    f"Merge column '{self.merge_column}' was not found in {name}."
                )
                valid = False
                continue

            if dataframe.columns.duplicated().any():
                self._error(f"{name} contains duplicate column names.")
                valid = False
                continue

            self._success(
                f"{name} is valid with {dataframe.shape[0]} row(s) and "
                f"{dataframe.shape[1]} column(s)."
            )

        if valid:
            self.dataframe_items = dataframe_items
            self._success("All input arguments passed validation.")
        else:
            self._error("One or more input arguments failed validation.")

        self.dataframe_items = dataframe_items

        return valid

    # ==========================================================
    # STEP 2: MERGING
    # ==========================================================

    def merge_data(self):
        """Merge all validated DataFrames and return the result."""
        self._header("MERGING DATAFRAMES")

        if not self.dataframe_items:
            self._error("There are no validated DataFrames available for merging.")
            return None

        first_name, merged_data = self.dataframe_items[0]
        merged_data = merged_data.copy()
        self._subheader(f"STARTING WITH: {first_name}")
        self._info(f"Initial rows: {merged_data.shape[0]}")
        self._info(f"Initial columns: {merged_data.shape[1]}")

        for name, dataframe in self.dataframe_items[1:]:
            self._subheader(f"MERGING WITH: {name}")

            duplicate_keys = dataframe[self.merge_column].duplicated().sum()
            if duplicate_keys:
                self._warning(
                    f"{name} contains {duplicate_keys} duplicate "
                    f"'{self.merge_column}' value(s); the merge may create extra rows."
                )
            else:
                self._success(f"{name} has unique '{self.merge_column}' values.")

            rows_before = len(merged_data)
            try:
                merged_data = pd.merge(
                    merged_data,
                    dataframe,
                    on=self.merge_column,
                    how=self.how,
                    suffixes=self.suffixes
                )
                self._info(f"Saving merged data after merging.")
                merged_data.to_csv("./data/processed/merged_full_data.csv", index=False)
                self._success(f"Saved merged data to 'processed/merged_full_data.csv'.")
            except Exception as error:
                self._error(f"Failed to merge with {name}: {error}")
                return None

            rows_after = len(merged_data)
            self._success(f"Merge with {name} completed.")
            self._info(f"Rows before merge: {rows_before}")
            self._info(f"Rows after merge: {rows_after}")

            if self.how == "inner" and rows_after < rows_before:
                self._warning(
                    f"{rows_before - rows_after} row(s) were removed by the INNER merge."
                )

        self._subheader("SORTING MERGED DATA")
        try:
            merged_data = merged_data.sort_values(
                by=self.merge_column
            ).reset_index(drop=True)
        except Exception as error:
            self._error(f"Could not sort by '{self.merge_column}': {error}")
            return None

        self.combined_data = merged_data
        self._success(f"Data sorted by '{self.merge_column}'.")
        self._subheader("MERGE COMPLETED")
        self._success(f"Final rows: {merged_data.shape[0]}")
        self._success(f"Final columns: {merged_data.shape[1]}")
        return self.combined_data

    # ==========================================================
    # FINAL REPORT AND ACCESSOR
    # ==========================================================

    def _final_report(self):
        """Display the final status of the merge operation."""
        self._header("FINAL DATA MERGE REPORT")
        if isinstance(self.dataframes, dict):
            dataframe_count = len(self.dataframes)
        elif isinstance(self.dataframes, (list, tuple)):
            dataframe_count = len(self.dataframes)
        else:
            dataframe_count = 0

        self._info(f"DataFrames provided: {dataframe_count}")
        self._info(f"Errors: {len(self.errors)}")
        self._info(f"Warnings: {len(self.warnings)}")

        if self.combined_data is not None:
            self._success(f"Final shape: {self.combined_data.shape}")
            self._info("Access combined data using merger.combined_data")
            self._info("Or use merger.get_data()")
        else:
            self._warning("No combined DataFrame is available.")

        self._header("DATA MERGING COMPLETE")

    def get_data(self):
        """Return the merged DataFrame, or None when merging failed."""
        return self.combined_data
