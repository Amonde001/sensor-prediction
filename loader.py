# loader.py used in main.py to load climate and moisture data. It is a reusable class for loading and merging CSV/Excel files, with features like argument validation, progress bar, and colored terminal messages.
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from colorama import Fore, Style, init


# Initialize colorama
init(autoreset=True)


class readData:
    """
    Data ingestion class for loading and merging CSV/Excel files.

    Features:
    - Validates input arguments
    - Finds all matching files in a directory
    - Loads CSV and Excel files
    - Supports custom column names
    - Processes a specified date/time column
    - Displays colored terminal messages
    - Displays a progress bar
    - Merges all DataFrames using INNER JOIN
    - Provides access to individual and combined DataFrames
    """

    # Supported file types
    SUPPORTED_FILE_TYPES = {
        "csv": [".csv"],
        "excel": [".xlsx", ".xls"],
        "xlsx": [".xlsx"],
        "xls": [".xls"]
    }

    def __init__(
        self,
        datapath,
        skiprows=0,
        column_names=None,
        date_column=None,
        file_type=None,
        sheet_name=None
    ):
        """
        Initialize the readData object.

        Parameters
        ----------
        datapath : str
            Directory containing the data files.

        skiprows : int, default=0
            Number of rows to skip when reading files.

        column_names : list, optional
            Custom column names to use when reading the files.

        date_column : str
            Name of the date/time column used for processing
            and merging.

        file_type : str
            File type to load. Supported:
            - csv
            - excel
            - xlsx
            - xls

        sheet_name : str, int, or None
            Excel sheet to load. If None, the default Excel sheet is used.
        """

        # Store input parameters
        self.datapath = Path(datapath)
        self.skiprows = skiprows
        self.column_names = column_names
        self.date_column = date_column
        self.sheet_name = sheet_name

        # Normalize file type
        self.file_type = (
            file_type.lower().strip()
            if isinstance(file_type, str)
            else file_type
        )

        # Store discovered files
        self.files_found = []

        # Store successfully loaded files
        self.loaded_files = []

        # Store failed files
        self.failed_files = []

        # Store individual DataFrames
        self.dataframes = {}

        # Store final merged DataFrame
        self.combined_data = None

        # Automatically run the complete system
        self._run_full_system()

    # ==========================================================
    # GENERAL DISPLAY METHODS
    # ==========================================================

    def _pause(self, seconds=0.2):
        """
        Pause execution for the specified number of seconds.
        """
        time.sleep(seconds)

    def _header(self, title):
        """
        Display a major section header.
        """
        print()
        print(Fore.CYAN + "-" * 70)
        print(Fore.CYAN + title.center(70))
        print(Fore.CYAN + "-" * 70)
        self._pause()

    def _subheader(self, title):
        """
        Display a subsection header.
        """
        print()
        print(Fore.YELLOW + "-" * 70)
        print(Fore.YELLOW + title)
        print(Fore.YELLOW + "-" * 70)
        self._pause()

    def _info(self, message):
        """
        Display an informational message.
        """
        print(Fore.WHITE + f"[INFO] {message}")
        self._pause()

    def _success(self, message):
        """
        Display a successful operation message.
        """
        print(Fore.GREEN + f"[SUCCESS] {message}")
        self._pause()

    def _warning(self, message):
        """
        Display a warning message.
        """
        print(Fore.YELLOW + f"[WARNING] {message}")
        self._pause()

    def _error(self, message):
        """
        Display an error message.
        """
        print(Fore.RED + f"[ERROR] {message}")
        self._pause()

    # ==========================================================
    # COMPLETE WORKFLOW
    # ==========================================================

    def _run_full_system(self):
        """
        Execute the complete data ingestion workflow automatically.
        """

        self._header("DATA INGESTION SYSTEM")

        self._info("Starting data ingestion process...")

        # ------------------------------------------------------
        # STEP 1: Validate arguments
        # ------------------------------------------------------
        if not self._validate_arguments():
            self._error("Validation failed. Data ingestion stopped.")
            return

        # ------------------------------------------------------
        # STEP 2: Find files
        # ------------------------------------------------------
        self.find_files()

        if not self.files_found:
            self._warning("No matching files were found.")
            self._final_report()
            return

        # ------------------------------------------------------
        # STEP 3: Load files
        # ------------------------------------------------------
        self.load_files()

        # ------------------------------------------------------
        # STEP 4: Merge files
        # ------------------------------------------------------
        if self.loaded_files:
            self.combine_data()
        else:
            self._warning("No files were successfully loaded.")

        # ------------------------------------------------------
        # STEP 5: Final report
        # ------------------------------------------------------
        self._final_report()

    # ==========================================================
    # STEP 1: VALIDATION
    # ==========================================================

    def _validate_arguments(self):
        """
        Validate all constructor arguments.
        """

        self._header("STEP 1: VALIDATING INPUTS")

        valid = True

        # ------------------------------------------------------
        # Validate directory
        # ------------------------------------------------------

        self._info(f"Checking data directory: {self.datapath}")

        if not self.datapath.exists():
            self._error(
                f"Directory does not exist: {self.datapath}"
            )
            valid = False

        elif not self.datapath.is_dir():
            self._error(
                f"Path is not a directory: {self.datapath}"
            )
            valid = False

        else:
            self._success("Data directory exists.")

        # ------------------------------------------------------
        # Validate skiprows
        # ------------------------------------------------------

        self._info(
            f"Checking skiprows value: {self.skiprows}"
        )

        if (
            not isinstance(self.skiprows, int)
            or isinstance(self.skiprows, bool)
            or self.skiprows < 0
        ):
            self._error(
                "skiprows must be a non-negative integer."
            )
            valid = False

        else:
            self._success("skiprows is valid.")

        # ------------------------------------------------------
        # Validate date column
        # ------------------------------------------------------

        self._info(
            f"Checking date column: {self.date_column}"
        )

        if not self.date_column:
            self._error(
                "date_column must be provided."
            )
            valid = False

        elif not isinstance(self.date_column, str):
            self._error(
                "date_column must be a string."
            )
            valid = False

        else:
            self._success(
                f"Date column '{self.date_column}' is valid."
            )

        # ------------------------------------------------------
        # Validate file type
        # ------------------------------------------------------

        self._info(
            f"Checking file type: {self.file_type}"
        )

        if self.file_type not in self.SUPPORTED_FILE_TYPES:
            self._error(
                "Unsupported file type. "
                "Use 'csv', 'excel', 'xlsx', or 'xls'."
            )
            valid = False

        else:
            self._success(
                f"File type '{self.file_type}' is supported."
            )

        # ------------------------------------------------------
        # Validate column names
        # ------------------------------------------------------

        if self.column_names is not None:

            self._info("Checking custom column names...")

            if not isinstance(self.column_names, list):
                self._error(
                    "column_names must be a list or None."
                )
                valid = False

            elif (
                not self.column_names
                or any(
                    not isinstance(column, str)
                    or not column.strip()
                    for column in self.column_names
                )
                or len(set(self.column_names)) != len(self.column_names)
            ):
                self._error(
                    "column_names must contain unique, non-empty strings."
                )
                valid = False

            else:
                self._success(
                    "Custom column names are valid."
                )

        else:
            self._info(
                "No custom column names provided. "
                "Using columns from files."
            )

        # ------------------------------------------------------
        # Validate sheet name
        # ------------------------------------------------------

        self._info(
            f"Checking sheet name: {self.sheet_name}"
        )

        if self.sheet_name is not None:

            if self.file_type == "csv":
                self._error(
                    "sheet_name can only be used with Excel files."
                )
                valid = False

            elif (
                isinstance(self.sheet_name, bool)
                or not isinstance(self.sheet_name, (str, int))
                or (
                    isinstance(self.sheet_name, str)
                    and not self.sheet_name.strip()
                )
                or (
                    isinstance(self.sheet_name, int)
                    and self.sheet_name < 0
                )
            ):
                self._error(
                    "sheet_name must be a non-empty sheet name, "
                    "a non-negative sheet index, or None."
                )
                valid = False

            else:
                self._success(
                    f"Excel sheet '{self.sheet_name}' is valid."
                )

        else:
            self._info(
                "No sheet_name provided. The first Excel sheet will be used."
            )

        # ------------------------------------------------------
        # Final validation status
        # ------------------------------------------------------

        if valid:
            self._success(
                "All input arguments passed validation."
            )
        else:
            self._error(
                "One or more input arguments failed validation."
            )

        return valid

    # ==========================================================
    # STEP 2: FIND FILES
    # ==========================================================

    def find_files(self):
        """
        Find all files matching the selected file type.
        """

        self._header("STEP 2: DISCOVERING DATA FILES")

        # Get valid extensions
        extensions = self.SUPPORTED_FILE_TYPES[self.file_type]

        self._info(
            f"Searching directory: {self.datapath}"
        )

        self._info(
            f"Accepted extensions: {extensions}"
        )

        # Search only direct files in the directory
        self.files_found = sorted(
            [
                file
                for file in self.datapath.iterdir()
                if file.is_file()
                and not file.name.startswith("~$")
                and file.suffix.lower() in extensions
            ]
        )

        # ------------------------------------------------------
        # No files
        # ------------------------------------------------------

        if not self.files_found:
            self._warning(
                "No matching files were found."
            )
            return

        # ------------------------------------------------------
        # Display number of files
        # ------------------------------------------------------

        self._success(
            f"Found {len(self.files_found)} file(s)."
        )

        # ------------------------------------------------------
        # Display every file
        # ------------------------------------------------------

        self._subheader("FILES FOUND")

        for number, filepath in enumerate(
            self.files_found,
            start=1
        ):

            print(
                Fore.WHITE
                + f"{number}. {filepath.name}"
            )

            self._pause()

    # ==========================================================
    # FILE READING
    # ==========================================================

    def _read_file(self, filepath):
        """
        Read a single CSV or Excel file.

        Parameters
        ----------
        filepath : Path
            Path to the file.

        Returns
        -------
        pandas.DataFrame
            Loaded DataFrame.
        """

        extension = filepath.suffix.lower()

        # ------------------------------------------------------
        # CSV
        # ------------------------------------------------------

        if extension == ".csv":

            if self.column_names is None:

                dataframe = pd.read_csv(
                    filepath,
                    skiprows=self.skiprows
                )

            else:

                dataframe = pd.read_csv(
                    filepath,
                    skiprows=self.skiprows,
                    names=self.column_names
                )

        # ------------------------------------------------------
        # Excel
        # ------------------------------------------------------

        elif extension in [".xlsx", ".xls"]:

            excel_options = {
                "skiprows": self.skiprows
            }

            if self.sheet_name is not None:
                excel_options["sheet_name"] = self.sheet_name

            if self.column_names is None:

                dataframe = pd.read_excel(
                    filepath,
                    **excel_options
                )

            else:

                dataframe = pd.read_excel(
                    filepath,
                    names=self.column_names,
                    **excel_options
                )

            if not isinstance(dataframe, pd.DataFrame):
                raise ValueError(
                    "Excel read returned multiple sheets. "
                    "Provide one sheet_name."
                )

        # ------------------------------------------------------
        # Unsupported extension
        # ------------------------------------------------------

        else:

            raise ValueError(
                f"Unsupported file extension: {extension}"
            )

        return dataframe

    # ==========================================================
    # DATE COLUMN PROCESSING
    # ==========================================================

    def _process_date_column(
        self,
        dataframe,
        filename
    ):
        """
        Convert the specified date column to pandas datetime.
        """

        # ------------------------------------------------------
        # Check whether date column exists
        # ------------------------------------------------------

        if self.date_column not in dataframe.columns:

            raise ValueError(
                f"Date column '{self.date_column}' "
                f"not found in {filename}."
            )

        self._info(
            f"Processing date column '{self.date_column}' "
            f"for {filename}..."
        )

        # ------------------------------------------------------
        # Convert to datetime
        # ------------------------------------------------------

        dataframe[self.date_column] = pd.to_datetime(
            dataframe[self.date_column],
            errors="coerce"
        )

        # ------------------------------------------------------
        # Check invalid dates
        # ------------------------------------------------------

        invalid_dates = dataframe[
            self.date_column
        ].isna().sum()

        if invalid_dates > 0:

            self._warning(
                f"{filename}: {invalid_dates} invalid "
                f"date value(s) converted to NaT."
            )

        else:

            self._success(
                f"{filename}: all date values converted successfully."
            )

        return dataframe

    # ==========================================================
    # STEP 3: LOAD FILES
    # ==========================================================

    def load_files(self):
        """
        Load all discovered files.

        A progress bar is displayed while files are loaded.
        The loader is intentionally kept visible for at least
        two seconds.
        """

        self._header("STEP 3: LOADING DATA FILES")

        self._info(
            f"Preparing to load {len(self.files_found)} file(s)..."
        )

        # Record starting time
        start_time = time.time()

        # ------------------------------------------------------
        # Progress bar
        # ------------------------------------------------------

        with tqdm(
            total=len(self.files_found),
            desc="Loading files",
            unit="file",
            ncols=100
        ) as progress:

            for filepath in self.files_found:

                self._subheader(
                    f"LOADING: {filepath.name}"
                )

                try:

                    # ------------------------------------------
                    # Read file
                    # ------------------------------------------

                    self._info(
                        f"Reading {filepath.name}..."
                    )

                    dataframe = self._read_file(filepath)

                    self._success(
                        f"{filepath.name} loaded successfully."
                    )

                    # ------------------------------------------
                    # Display shape
                    # ------------------------------------------

                    self._info(
                        f"Rows: {dataframe.shape[0]}"
                    )

                    self._info(
                        f"Columns: {dataframe.shape[1]}"
                    )

                    # ------------------------------------------
                    # Process date column
                    # ------------------------------------------

                    dataframe = self._process_date_column(
                        dataframe,
                        filepath.name
                    )

                    # ------------------------------------------
                    # Store DataFrame
                    # ------------------------------------------

                    self.dataframes[
                        filepath.name
                    ] = dataframe

                    # ------------------------------------------
                    # Record successful file
                    # ------------------------------------------

                    self.loaded_files.append(
                        filepath.name
                    )

                    self._success(
                        f"{filepath.name} added to dataframes."
                    )

                except Exception as error:

                    # ------------------------------------------
                    # Record failed file
                    # ------------------------------------------

                    self.failed_files.append(
                        {
                            "file": filepath.name,
                            "error": str(error)
                        }
                    )

                    self._error(
                        f"Failed to load {filepath.name}: "
                        f"{error}"
                    )

                # Update progress bar
                progress.update(1)

        # ------------------------------------------------------
        # Ensure progress system remains visible for at least
        # two seconds
        # ------------------------------------------------------

        elapsed_time = time.time() - start_time

        if elapsed_time < 2:
            time.sleep(2 - elapsed_time)

        # ------------------------------------------------------
        # Loading summary
        # ------------------------------------------------------

        self._subheader("LOADING SUMMARY")

        self._success(
            f"Successfully loaded: "
            f"{len(self.loaded_files)} file(s)"
        )

        if self.failed_files:

            self._warning(
                f"Failed to load: "
                f"{len(self.failed_files)} file(s)"
            )

        else:

            self._success(
                "All discovered files were loaded successfully."
            )

    # ==========================================================
    # STEP 4: MERGE DATA
    # ==========================================================

    def combine_data(self):
        """
        Merge all loaded DataFrames using the date column.

        Merge method:
            INNER JOIN

        Example:

            DataFrame 1:
                valid_time | t2m

            DataFrame 2:
                valid_time | d2m

            DataFrame 3:
                valid_time | sp

        Result:

            valid_time | t2m | d2m | sp

        Only valid_time values that exist in ALL DataFrames
        are retained.
        """

        self._header("STEP 4: MERGING DATA")

        # ------------------------------------------------------
        # Check whether DataFrames exist
        # ------------------------------------------------------

        if not self.dataframes:

            self._warning(
                "There are no DataFrames available for merging."
            )

            return None

        self._info(
            f"Number of DataFrames to merge: "
            f"{len(self.dataframes)}"
        )

        self._info(
            f"Merge column: '{self.date_column}'"
        )

        self._info(
            "Merge type: INNER JOIN"
        )

        # ------------------------------------------------------
        # Convert dictionary to list
        # ------------------------------------------------------

        dataframe_items = list(
            self.dataframes.items()
        )

        # ------------------------------------------------------
        # Start with the first DataFrame
        # ------------------------------------------------------

        first_name, merged_data = dataframe_items[0]

        self._subheader(
            f"STARTING MERGE WITH: {first_name}"
        )

        self._info(
            f"Initial rows: {merged_data.shape[0]}"
        )

        self._info(
            f"Initial columns: {merged_data.shape[1]}"
        )

        # ------------------------------------------------------
        # Merge remaining DataFrames one at a time
        # ------------------------------------------------------

        for filename, dataframe in dataframe_items[1:]:

            self._subheader(
                f"MERGING WITH: {filename}"
            )

            # ----------------------------------------------
            # Check merge column
            # ----------------------------------------------

            if self.date_column not in dataframe.columns:

                self._error(
                    f"'{self.date_column}' not found in "
                    f"{filename}. Skipping this file."
                )

                continue

            # ----------------------------------------------
            # Check duplicate dates
            # ----------------------------------------------

            duplicate_dates = (
                dataframe[self.date_column]
                .duplicated()
                .sum()
            )

            if duplicate_dates > 0:

                self._warning(
                    f"{filename} contains "
                    f"{duplicate_dates} duplicate "
                    f"'{self.date_column}' values."
                )

                self._warning(
                    "Duplicate timestamps may create "
                    "multiple rows during the merge."
                )

            else:

                self._success(
                    f"{filename} has unique "
                    f"'{self.date_column}' values."
                )

            # ----------------------------------------------
            # Store rows before merge
            # ----------------------------------------------

            rows_before = merged_data.shape[0]

            # ----------------------------------------------
            # Perform INNER JOIN
            # ----------------------------------------------

            merged_data = pd.merge(
                merged_data,
                dataframe,
                on=self.date_column,
                how="inner",
                suffixes=("", "_duplicate")
            )

            # ----------------------------------------------
            # Store rows after merge
            # ----------------------------------------------

            rows_after = merged_data.shape[0]

            rows_removed = (
                rows_before - rows_after
            )

            self._success(
                f"Merge with {filename} completed."
            )

            self._info(
                f"Rows before merge: {rows_before}"
            )

            self._info(
                f"Rows after merge: {rows_after}"
            )

            self._info(
                f"Rows removed by INNER JOIN: {rows_removed}"
            )

        # ======================================================
        # REMOVE DUPLICATE COLUMNS CREATED BY MERGING
        # ======================================================

        duplicate_columns = [
            column
            for column in merged_data.columns
            if column.endswith("_duplicate")
        ]

        if duplicate_columns:

            self._subheader(
                "REMOVING DUPLICATE COLUMNS"
            )

            self._info(
                f"Duplicate columns found: "
                f"{len(duplicate_columns)}"
            )

            merged_data = merged_data.drop(
                columns=duplicate_columns
            )

            self._success(
                "Duplicate columns removed."
            )

        else:

            self._info(
                "No duplicate columns were created."
            )

        # ======================================================
        # SORT BY DATE
        # ======================================================

        self._subheader(
            "SORTING MERGED DATA"
        )

        merged_data = (
            merged_data
            .sort_values(
                by=self.date_column
            )
            .reset_index(drop=True)
        )

        self._success(
            f"Data sorted by '{self.date_column}'."
        )

        # ======================================================
        # STORE FINAL DATA
        # ======================================================

        self.combined_data = merged_data

        # ======================================================
        # FINAL MERGE INFORMATION
        # ======================================================

        self._subheader(
            "MERGE COMPLETED"
        )

        self._success(
            f"Final rows: {merged_data.shape[0]}"
        )

        self._success(
            f"Final columns: {merged_data.shape[1]}"
        )

        self._success(
            "Final DataFrame stored in "
            "reader.combined_data"
        )

        return self.combined_data

    # ==========================================================
    # FINAL REPORT
    # ==========================================================

    def _final_report(self):
        """
        Display the final status of the data ingestion process.
        """

        self._header("FINAL DATA INGESTION REPORT")

        # ------------------------------------------------------
        # File statistics
        # ------------------------------------------------------

        self._info(
            f"Files discovered : {len(self.files_found)}"
        )

        self._info(
            f"Files loaded     : {len(self.loaded_files)}"
        )

        self._info(
            f"Files failed     : {len(self.failed_files)}"
        )

        # ------------------------------------------------------
        # Success rate
        # ------------------------------------------------------

        if self.files_found:

            success_rate = (
                len(self.loaded_files)
                / len(self.files_found)
            ) * 100

            self._info(
                f"Loading success rate: "
                f"{success_rate:.2f}%"
            )

        # ------------------------------------------------------
        # Failed files
        # ------------------------------------------------------

        if self.failed_files:

            self._subheader(
                "FAILED FILES"
            )

            for failure in self.failed_files:

                self._error(
                    f"{failure['file']}: "
                    f"{failure['error']}"
                )

        # ------------------------------------------------------
        # Individual DataFrames
        # ------------------------------------------------------

        if self.dataframes:

            self._subheader(
                "AVAILABLE INDIVIDUAL DATAFRAMES"
            )

            for filename, dataframe in (
                self.dataframes.items()
            ):

                print(
                    Fore.WHITE
                    + f"reader.dataframes['{filename}']"
                )

                self._info(
                    f"Shape: {dataframe.shape}"
                )

            # --------------------------------------------------
            # Combined DataFrame
            # --------------------------------------------------

            if self.combined_data is not None:

                self._subheader(
                    "FINAL COMBINED DATA"
                )

                self._success(
                    "Access combined data using:"
                )

                print(
                    Fore.GREEN
                    + "reader.combined_data"
                )

                self._info(
                    "Or use:"
                )

                print(
                    Fore.GREEN
                    + "reader.get_data()"
                )

                self._info(
                    f"Final shape: "
                    f"{self.combined_data.shape}"
                )

        # ------------------------------------------------------
        # End
        # ------------------------------------------------------

        self._header(
            "DATA INGESTION COMPLETE"
        )

        self._success(
            "The readData system has finished."
        )

    # ==========================================================
    # GET COMBINED DATA
    # ==========================================================

    def get_data(self):
        """
        Return the final merged DataFrame.

        Returns
        -------
        pandas.DataFrame or None
        """

        return self.combined_data

    # ==========================================================
    # DISPLAY DATA INFORMATION
    # ==========================================================

    def info(self):
        """
        Display information about all loaded DataFrames.
        """

        self._header("DATAFRAME INFORMATION")

        if not self.dataframes:

            self._warning(
                "No DataFrames are currently loaded."
            )

            return

        for filename, dataframe in (
            self.dataframes.items()
        ):

            self._subheader(
                filename
            )

            self._info(
                f"Rows: {dataframe.shape[0]}"
            )

            self._info(
                f"Columns: {dataframe.shape[1]}"
            )

            print(
                Fore.WHITE
                + "Columns:"
            )

            for column in dataframe.columns:

                print(
                    Fore.WHITE
                    + f"  - {column}"
                )

                self._pause()

        # ------------------------------------------------------
        # Combined data information
        # ------------------------------------------------------

        if self.combined_data is not None:

            self._subheader(
                "COMBINED DATA"
            )

            self._info(
                f"Rows: {self.combined_data.shape[0]}"
            )

            self._info(
                f"Columns: {self.combined_data.shape[1]}"
            )

            print(
                Fore.WHITE
                + "Columns:"
            )

            for column in self.combined_data.columns:

                print(
                    Fore.WHITE
                    + f"  - {column}"
                )

                self._pause()
