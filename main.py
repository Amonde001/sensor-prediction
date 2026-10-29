# main.py for the moisture sensor data pipeline. It orchestrates the loading and merging of climate and moisture data, providing a clear workflow with colored terminal messages for better visibility.
from pathlib import Path

from colorama import Fore, init

from loader import readData
from merger import merger


# Initialize colorama
init(autoreset=True)


BASE_DIR = Path(__file__).resolve().parent
CLIMATE_DATA_PATH = BASE_DIR / "data" / "climate"
MOISTURE_DATA_PATH = BASE_DIR / "data" / "moisture"
MOISTURE_FILE = MOISTURE_DATA_PATH / "raw.xlsx"


def print_step(message):
    """Print a main-workflow step using the project display style."""
    print()
    print(Fore.CYAN + "=" * 70)
    print(Fore.CYAN + message.center(70))
    print(Fore.CYAN + "=" * 70)


def run_pipeline():
    """Load climate and moisture data, then merge the loaded results."""
    print_step("MOISTURE SENSOR DATA PIPELINE")

    print_step("STEP 1: LOADING CLIMATE DATA")
    climate_loader = readData(
        datapath=CLIMATE_DATA_PATH,
        skiprows=0,
        date_column="valid_time",
        file_type="csv"
    )

    if climate_loader.combined_data is None:
        print(Fore.RED + "[ERROR] Climate data could not be loaded.")
        return None

    print_step("STEP 2: LOADING MOISTURE DATA")
    if not MOISTURE_FILE.exists():
        print(Fore.RED + f"[ERROR] Moisture file does not exist: {MOISTURE_FILE}")
        return None

    moisture_loader = readData(
        datapath=MOISTURE_DATA_PATH,
        skiprows=3,
        date_column="valid_time",
        file_type="xlsx",
        column_names=[
            "valid_time",
            "Port 1",
            "Port 2",
            "Port 3",
            "Port 4",
            "Port 5"
        ],
        sheet_name="Unprocessed Data"
    )

    if moisture_loader.combined_data is None:
        print(Fore.RED + "[ERROR] Moisture data could not be loaded.")
        return None

    print_step("STEP 3: MERGING CLIMATE AND MOISTURE DATA")
    data_merger = merger(
        dataframes={
            "Climate data": climate_loader.combined_data,
            "Moisture data": moisture_loader.combined_data
        },
        merge_column="valid_time",
        how="inner"
    )

    if data_merger.combined_data is None:
        print(Fore.RED + "[ERROR] Climate and moisture data could not be merged.")
        return None

    print_step("STEP 4: PIPELINE COMPLETE")
    print(Fore.GREEN + f"[SUCCESS] Final shape: {data_merger.combined_data.shape}")
    print(Fore.GREEN + "[SUCCESS] Merged data is available in the returned DataFrame.")

    return data_merger.combined_data


def main():
    """Run the complete loading and merging workflow."""
    try:
        return run_pipeline()
    except Exception as error:
        print(Fore.RED + f"[ERROR] Unexpected pipeline failure: {error}")
        return None


if __name__ == "__main__":
    main()
