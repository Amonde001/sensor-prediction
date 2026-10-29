# dataset_processor.py used in climate_moisture_feature_pipieline.py to load, process, clean, and analyze climate data. 
# It is a reusable class for loading and merging CSV/Excel files, with features like argument validation, progress bar
"""Reusable classes for loading, processing, cleaning, and analyzing climate data."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


class LoadData:
    """Load the ERA5 CSV files in a directory and expose named datasets."""

    FILES = {
        "temperature_dewpoint": "reanalysis-era5-land-timeseries-sfc-2m-temperatureknkrsipg.csv",
        "pressure_precipitation": "reanalysis-era5-land-timeseries-sfc-pressure-precipitation5_u_5abt.csv",
        "radiation": "reanalysis-era5-land-timeseries-sfc-radiation-heatcs5zfi6f.csv",
        "surface_temperature": "reanalysis-era5-land-timeseries-sfc-skin-temperature_fxobkys.csv",
        "snow_depth": "reanalysis-era5-land-timeseries-sfc-snowes2undel.csv",
        "soil_temperature": "reanalysis-era5-land-timeseries-sfc-soil-temperaturetfqpw8qf.csv",
        "soil_water": "reanalysis-era5-land-timeseries-sfc-soil-water6s96577b.csv",
        "wind": "reanalysis-era5-land-timeseries-sfc-windgk1ro2q8.csv",
    }

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.dfs = self.load_csv()

    def load_csv(self):
        """Read every CSV file in ``data_dir`` into a name-keyed dictionary."""
        return {
            file.name: pd.read_csv(file)
            for file in self.data_dir.glob("*.csv")
        }

    def show(self):
        """Print the shape and first rows of each loaded dataset."""
        for name, dataframe in self.dfs.items():
            print(f"{name}: {dataframe.shape}")
            print(dataframe.head())
            print()

    def _get(self, key):
        return self.dfs[self.FILES[key]]

    def temperature_dewpoint_above_surface(self):
        return self._get("temperature_dewpoint")

    def surface_pressure_precipitation(self):
        return self._get("pressure_precipitation")

    def surface_radiation(self):
        return self._get("radiation")

    def surface_temperature(self):
        return self._get("surface_temperature")

    def snow_depth(self):
        return self._get("snow_depth")

    def soil_temperature(self):
        return self._get("soil_temperature")

    def soil_water(self):
        return self._get("soil_water")

    def soil_wind(self):
        return self._get("wind")


class Info:
    """Print basic information for each dataset provided by ``LoadData``."""

    def __init__(self, dataframes):
        self.dataframes = dataframes

    def get_info(self):
        datasets = {
            "Temperature and Dewpoint above the Surface": self.dataframes.temperature_dewpoint_above_surface(),
            "Surface Pressure Precipitation": self.dataframes.surface_pressure_precipitation(),
            "Surface Heat Radiation": self.dataframes.surface_radiation(),
            "Surface Temperature": self.dataframes.surface_temperature(),
            "Snow Depth": self.dataframes.snow_depth(),
            "Soil Temperature": self.dataframes.soil_temperature(),
            "Soil Wind": self.dataframes.soil_wind(),
            "Soil Water": self.dataframes.soil_water(),
        }

        for name, dataframe in datasets.items():
            print("=" * 60)
            print(name)
            print("-" * 60)
            print(f"Shape: {dataframe.shape}")
            print(f"Columns: {dataframe.columns.to_list()}\n")
            print("Head")
            print("-" * 60)
            print(dataframe.head(3))
            print("\nInfo")
            print("-" * 60)
            dataframe.info()
            print()


class Processor:
    """Merge ERA5 datasets and replace short variable names with readable names."""

    MERGE_KEYS = ["valid_time", "latitude", "longitude"]

    def __init__(self, dataframes, era5_dictionary):
        self.dataframes = dataframes
        self.era5_dictionary = era5_dictionary

    def merge(self):
        """Inner-join all climate datasets on time and grid coordinates."""
        frames = [
            self.dataframes.temperature_dewpoint_above_surface(),
            self.dataframes.surface_pressure_precipitation(),
            self.dataframes.surface_radiation(),
            self.dataframes.surface_temperature(),
            self.dataframes.snow_depth(),
            self.dataframes.soil_temperature(),
            self.dataframes.soil_wind(),
            self.dataframes.soil_water(),
        ]
        normalized_frames = []
        for dataframe in frames:
            normalized = dataframe.copy()
            normalized["valid_time"] = pd.to_datetime(
                normalized["valid_time"], utc=True, errors="coerce"
            ).dt.floor("h")
            normalized["latitude"] = normalized["latitude"].round(6)
            normalized["longitude"] = normalized["longitude"].round(6)
            normalized_frames.append(normalized)

        merged = normalized_frames[0]
        for dataframe in normalized_frames[1:]:
            merged = pd.merge(merged, dataframe, on=self.MERGE_KEYS, how="inner")
        return merged

    def process_columns(self, dataframe):
        """Rename ERA5 short variable codes using the supplied metadata mapping."""
        mapping = {
            column: metadata["full_name"]
            for column, metadata in self.era5_dictionary.items()
            if column in dataframe.columns
        }
        return dataframe.rename(columns=mapping)


class Cleaner:
    """Inspect and clean a climate dataframe without changing the input object."""

    def __init__(self, dataframe):
        self.df = dataframe.copy()

    def basic_info(self):
        print("=" * 70)
        print("BASIC DATASET INFORMATION")
        print("=" * 70)
        print(f"Rows    : {self.df.shape[0]:,}")
        print(f"Columns : {self.df.shape[1]:,}\n")
        print("Columns:")
        print(self.df.columns.tolist())
        print("\nData types:")
        print(self.df.dtypes)

    def check_duplicates(self):
        duplicates = self.df.duplicated().sum()
        print(f"Duplicate rows: {duplicates:,}")
        return duplicates

    def check_duplicate_observations(self):
        columns = ["valid_time", "latitude", "longitude"]
        missing = [column for column in columns if column not in self.df.columns]
        if missing:
            print(f"Missing columns: {missing}")
            return None
        duplicates = self.df.duplicated(subset=columns).sum()
        print(f"Duplicate time-location observations: {duplicates:,}")
        return duplicates

    def check_data_types(self):
        print(self.df.dtypes)

    def process_datetime(self):
        if "valid_time" not in self.df.columns:
            print("'valid_time' column not found.")
            return
        self.df["valid_time"] = pd.to_datetime(self.df["valid_time"], errors="coerce")
        print(f"Minimum time: {self.df['valid_time'].min()}")
        print(f"Maximum time: {self.df['valid_time'].max()}")
        print(f"Invalid datetime values: {self.df['valid_time'].isna().sum():,}")

    def check_time_order(self):
        if "valid_time" in self.df.columns:
            print(f"Chronologically ordered: {self.df['valid_time'].is_monotonic_increasing}")

    def check_missing(self):
        result = pd.DataFrame({
            "Missing": self.df.isna().sum(),
            "Percentage": (self.df.isna().mean() * 100).round(2),
        })
        result = result[result["Missing"] > 0].sort_values("Missing", ascending=False)
        print("No missing values found." if result.empty else result)
        return result

    def check_infinite(self):
        numeric = self.df.select_dtypes(include=np.number).columns
        result = np.isinf(self.df[numeric]).sum()
        result = result[result > 0]
        print("No infinite values found." if result.empty else result)
        return result

    def check_zeros(self):
        numeric = self.df.select_dtypes(include=np.number).columns
        result = (self.df[numeric] == 0).sum()
        result = result[result > 0].sort_values(ascending=False)
        print(result)
        return result

    def clean_surface_pressure(self):
        column = "surface_pressure"
        if column not in self.df.columns:
            print(f"'{column}' not found.")
            return
        zero_count = (self.df[column] == 0).sum()
        self.df[column] = self.df[column].replace(0, np.nan)
        print(f"{zero_count:,} zero values in {column} converted to NaN.")

    def check_constant_columns(self):
        numeric = self.df.select_dtypes(include=np.number).columns
        result = [column for column in numeric if self.df[column].nunique(dropna=True) <= 1]
        print(result if result else "No constant numerical columns found.")
        return result

    def check_physical_ranges(self):
        results = {}
        if "latitude" in self.df:
            results["latitude"] = ((self.df["latitude"] < -90) | (self.df["latitude"] > 90)).sum()
        if "longitude" in self.df:
            results["longitude"] = ((self.df["longitude"] < -180) | (self.df["longitude"] > 180)).sum()
        for column in [
            "volumetric_soil_water_level_1", "volumetric_soil_water_level_2",
            "volumetric_soil_water_level_3", "volumetric_soil_water_level_4", "snowc",
        ]:
            if column in self.df:
                results[column] = ((self.df[column] < 0) | (self.df[column] > 1)).sum()
        result = pd.Series(results)
        print(result)
        return result

    def summary_statistics(self):
        return self.df.describe().T

    def check_outliers(self):
        results = {}
        for column in self.df.select_dtypes(include=np.number):
            series = self.df[column].dropna()
            if series.empty:
                continue
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            results[column] = {"Q1": q1, "Q3": q3, "IQR": iqr, "Lower": lower, "Upper": upper,
                               "Outliers": ((series < lower) | (series > upper)).sum()}
        result = pd.DataFrame(results).T
        print(result)
        return result

    def correlation_matrix(self):
        return self.df.select_dtypes(include=np.number).corr()

    def check_high_correlation(self, threshold=0.90):
        numeric = self.df.select_dtypes(include=np.number)
        correlation = numeric.corr().abs()
        upper = correlation.where(np.triu(np.ones(correlation.shape, dtype=bool), k=1))
        result = upper.stack().reset_index()
        result.columns = ["Variable 1", "Variable 2", "Correlation"]
        result = result[result["Correlation"] >= threshold].sort_values("Correlation", ascending=False)
        print(result)
        return result

    def create_time_features(self):
        if "valid_time" not in self.df.columns:
            print("'valid_time' not found.")
            return
        self.df["hour"] = self.df["valid_time"].dt.hour
        self.df["day"] = self.df["valid_time"].dt.day
        self.df["month"] = self.df["valid_time"].dt.month
        self.df["day_of_week"] = self.df["valid_time"].dt.dayofweek
        self.df["day_of_year"] = self.df["valid_time"].dt.dayofyear

    def create_wind_speed(self):
        u, v = "10m_u_component_of_wind", "10m_v_component_of_wind"
        if u not in self.df or v not in self.df:
            print("Wind component columns not found.")
            return
        self.df["wind_speed"] = np.sqrt(self.df[u] ** 2 + self.df[v] ** 2)

    def create_dewpoint_depression(self):
        temperature, dewpoint = "2_m_air_temperature", "2m_dewpoint_temperature"
        if temperature not in self.df or dewpoint not in self.df:
            print("Temperature or dewpoint column not found.")
            return
        self.df["dewpoint_depression"] = self.df[temperature] - self.df[dewpoint]

    def target_summary(self, target_column):
        if target_column not in self.df:
            print(f"Target column '{target_column}' not found.")
            return
        target = self.df[target_column]
        print(f"Target: {target_column}\nMissing: {target.isna().sum():,}\nUnique values: {target.nunique():,}")
        print(target.describe())

    def remove_duplicates(self):
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        print(f"Removed {before - len(self.df):,} duplicate rows.")

    def sort_data(self):
        columns = [column for column in ["valid_time", "latitude", "longitude"] if column in self.df]
        if columns:
            self.df = self.df.sort_values(columns).reset_index(drop=True)
            print(f"Data sorted by {columns}.")

    def replace_infinite(self):
        numeric = self.df.select_dtypes(include=np.number).columns
        self.df[numeric] = self.df[numeric].replace([np.inf, -np.inf], np.nan)

    def get_data(self):
        return self.df.copy()


class FeatureAnalysis:
    """Run descriptive, correlation, time-series, and feature-selection analyses."""

    def __init__(self, df, target=None, datetime_column="valid_time"):
        self.df = df.copy()
        self.target = target
        self.datetime_column = datetime_column
        self.results = {}
        self.drop_candidates = set()
        self.high_correlation_pairs = []

    def basic_info(self):
        print(f"Rows: {self.df.shape[0]:,}\nColumns: {self.df.shape[1]:,}")
        print(self.df.dtypes)
        print(self.df.columns.tolist())

    def descriptive_statistics(self):
        numeric = self.df.select_dtypes(include=np.number)
        result = numeric.describe().T
        result["missing"] = numeric.isna().sum()
        result["missing_percentage"] = numeric.isna().mean() * 100
        result["unique_values"] = numeric.nunique()
        self.results["descriptive_statistics"] = result
        print(result)
        return result

    def missing_values(self):
        result = pd.DataFrame({"missing_count": self.df.isna().sum(), "missing_percentage": self.df.isna().mean() * 100})
        result = result.sort_values("missing_percentage", ascending=False)
        self.results["missing_values"] = result
        print(result)
        return result

    def unique_value_analysis(self):
        result = pd.DataFrame({"unique_values": self.df.nunique(), "unique_percentage": self.df.nunique() / len(self.df) * 100})
        result = result.sort_values("unique_values")
        self.results["unique_values"] = result
        print(result)
        return result

    def constant_features(self):
        result = [column for column in self.df.select_dtypes(include=np.number) if self.df[column].nunique(dropna=True) <= 1]
        self.drop_candidates.update(result)
        print(result)
        return result

    def near_constant_features(self, threshold=0.99):
        candidates = []
        for column in self.df.select_dtypes(include=np.number):
            frequencies = self.df[column].value_counts(normalize=True, dropna=False)
            if not frequencies.empty and frequencies.iloc[0] >= threshold:
                candidates.append({"column": column, "dominant_percentage": frequencies.iloc[0] * 100})
                self.drop_candidates.add(column)
        result = pd.DataFrame(candidates)
        self.results["near_constant"] = result
        print(result)
        return result

    def variance_analysis(self):
        numeric = self.df.select_dtypes(include=np.number)
        result = pd.DataFrame({"variance": numeric.var(), "standard_deviation": numeric.std(), "minimum": numeric.min(), "maximum": numeric.max()}).sort_values("variance")
        self.results["variance"] = result
        print(result)
        return result

    def histogram(self, column, bins=30):
        if column not in self.df:
            print(f"Column '{column}' not found.")
            return
        sns.histplot(data=self.df, x=column, bins=bins, kde=True)
        plt.title(f"Distribution of {column}")
        plt.tight_layout()
        plt.show()

    def all_histograms(self, columns=None, bins=30):
        columns = columns or self.df.select_dtypes(include=np.number).columns.tolist()
        for column in columns:
            self.histogram(column, bins)

    def boxplot(self, column):
        if column not in self.df:
            print(f"Column '{column}' not found.")
            return
        sns.boxplot(x=self.df[column])
        plt.title(f"Boxplot of {column}")
        plt.tight_layout()
        plt.show()

    def all_boxplots(self, columns=None):
        columns = columns or self.df.select_dtypes(include=np.number).columns.tolist()
        for column in columns:
            self.boxplot(column)

    def correlation_matrix(self, method="pearson"):
        result = self.df.select_dtypes(include=np.number).corr(method=method)
        self.results["correlation"] = result
        sns.heatmap(result, cmap="coolwarm", center=0)
        plt.title(f"{method.capitalize()} Correlation Matrix")
        plt.tight_layout()
        plt.show()
        return result

    def highly_correlated_features(self, threshold=0.90):
        correlation = self.df.select_dtypes(include=np.number).corr().abs()
        pairs = [{"feature_1": columns[i], "feature_2": columns[j], "correlation": correlation.iloc[i, j]}
                 for columns in [correlation.columns]
                 for i in range(len(columns)) for j in range(i + 1, len(columns))
                 if correlation.iloc[i, j] >= threshold]
        result = pd.DataFrame(pairs)
        self.high_correlation_pairs = pairs
        self.results["high_correlation"] = result
        print(result)
        return result

    def target_correlation(self, method="pearson"):
        targets = [self.target] if isinstance(self.target, str) else list(self.target or [])
        targets = [target for target in targets if target in self.df.columns]
        if not targets:
            print("A valid target variable is required.")
            return None

        result = self.df.select_dtypes(include=np.number).corr(method=method)[targets]
        if len(targets) == 1:
            result = result[targets[0]].sort_values(ascending=False)
        self.results["target_correlation"] = result
        print(result)
        return result

    def target_comparison(self, feature):
        if self.target is None or feature not in self.df or self.target not in self.df:
            print("The target and feature must exist in the dataframe.")
            return
        sns.scatterplot(data=self.df, x=feature, y=self.target, alpha=0.4)
        plt.title(f"{feature} vs {self.target}")
        plt.tight_layout()
        plt.show()

    def mutual_information(self):
        targets = [self.target] if isinstance(self.target, str) else list(self.target or [])
        targets = [target for target in targets if target in self.df.columns]
        if not targets:
            print("A valid target variable is required.")
            return None

        numeric = self.df.select_dtypes(include=np.number)
        X = numeric.drop(columns=targets)
        if X.empty:
            print("No numeric predictor columns available.")
            return None

        results = {}

        for target in targets:
            y = numeric[target]
            valid = X.notna().all(axis=1) & y.notna()
            if valid.any():
                results[target] = pd.Series(
                    mutual_info_regression(X.loc[valid], y.loc[valid], random_state=42),
                    index=X.columns,
                )

        if not results:
            print("No valid observations available.")
            return None

        result = pd.DataFrame(results)
        if len(targets) == 1:
            result = result[targets[0]].sort_values(ascending=False)
        self.results["mutual_information"] = result
        print(result)
        return result

    def time_series_analysis(self, columns=None):
        """Plot one or more columns together against the datetime column."""
        if self.datetime_column not in self.df:
            print("Datetime or data column not found.")
            return

        if columns is None:
            columns = [
                column for column in self.df.select_dtypes(include=np.number).columns
                if column != self.target
            ]
        elif isinstance(columns, str):
            columns = [columns]

        columns = [column for column in columns if column in self.df.columns]
        if not columns:
            print("No valid data columns found.")
            return

        data = self.df[[self.datetime_column] + columns].dropna().sort_values(self.datetime_column)
        plt.figure(figsize=(15, 6))
        for column in columns:
            plt.plot(data[self.datetime_column], data[column], label=column)

        plt.title("Time Series")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def time_variability(self, columns=None):
        """Plot monthly mean variability for one or more columns together."""
        if self.datetime_column not in self.df:
            print("Datetime or data column not found.")
            return

        if columns is None:
            columns = [column for column in self.df.select_dtypes(include=np.number).columns]
        elif isinstance(columns, str):
            columns = [columns]
        columns = [column for column in columns if column in self.df.columns]
        if not columns:
            print("No valid data columns found.")
            return

        data = self.df.copy()
        data["analysis_month"] = data[self.datetime_column].dt.month
        result = data.groupby("analysis_month")[columns].mean()

        plt.figure(figsize=(12, 6))
        for column in columns:
            plt.plot(result.index, result[column], marker="o", label=column)

        plt.title("Monthly Mean Variability")
        plt.xlabel("Month")
        plt.ylabel("Mean value")
        plt.legend()
        plt.tight_layout()
        plt.show()
        return result

    def seasonal_analysis(self, columns=None):
        """Plot monthly seasonal means for one or more columns together."""
        if self.datetime_column not in self.df:
            print("Datetime or data column not found.")
            return

        if columns is None:
            columns = [column for column in self.df.select_dtypes(include=np.number).columns]
        elif isinstance(columns, str):
            columns = [columns]
        columns = [column for column in columns if column in self.df.columns]
        if not columns:
            print("No valid data columns found.")
            return

        data = self.df.copy()
        data["month"] = data[self.datetime_column].dt.month
        result = data.groupby("month")[columns].mean()

        plt.figure(figsize=(12, 6))
        for column in columns:
            plt.plot(result.index, result[column], marker="o", label=column)

        plt.title("Seasonal Monthly Means")
        plt.xlabel("Month")
        plt.ylabel("Mean value")
        plt.xticks(range(1, 13))
        plt.legend()
        plt.tight_layout()
        plt.show()
        return result

    def vif_analysis(self, max_features=15):
        numeric = self.df.select_dtypes(include=np.number).copy()
        targets = [self.target] if isinstance(self.target, str) else list(self.target or [])
        numeric = numeric.drop(columns=[target for target in targets if target in numeric], errors="ignore")
        numeric = numeric.loc[:, numeric.nunique() > 1].fillna(numeric.median())
        numeric = numeric.iloc[:, :max_features]
        result = pd.DataFrame({"feature": numeric.columns, "VIF": [variance_inflation_factor(numeric.values, i) for i in range(numeric.shape[1])]})
        result = result.sort_values("VIF", ascending=False)
        self.results["vif"] = result
        return result

    def recommend_drop_candidates(self, near_constant_threshold=0.99, correlation_threshold=0.95):
        self.constant_features()
        self.near_constant_features(near_constant_threshold)
        self.highly_correlated_features(correlation_threshold)
        return self.get_drop_candidates()

    def drop_unnecessary_columns(
        self,
        correlation_threshold=0.95,
        protected_columns=None,
        columns_to_drop=None,
    ):
        """Drop explicit unwanted columns and redundant correlated columns.

        The second feature in each highly correlated pair is removed unless it
        is listed in ``protected_columns``. ``columns_to_drop`` can be used for
        known unwanted fields such as coordinates, snow fields, or pressure.
        """
        protected = set(protected_columns or [])
        columns_to_remove = set(columns_to_drop or [])
        self.highly_correlated_features(correlation_threshold)

        for pair in self.high_correlation_pairs:
            feature_1 = pair["feature_1"]
            feature_2 = pair["feature_2"]

            if feature_1 in protected and feature_2 not in protected:
                columns_to_remove.add(feature_2)
            elif feature_2 in protected and feature_1 not in protected:
                columns_to_remove.add(feature_1)
            elif feature_1 not in protected and feature_2 not in protected:
                columns_to_remove.add(feature_2)

        columns_to_remove.difference_update(protected)
        columns_to_remove = sorted(columns_to_remove.intersection(self.df.columns))
        self.df = self.df.drop(columns=columns_to_remove)
        self.drop_candidates.update(columns_to_remove)

        print(f"Dropped {len(columns_to_remove)} unnecessary columns.")
        print(f"Columns dropped: {columns_to_remove}")
        return columns_to_remove

    def get_data(self):
        """Return the current analysis dataframe."""
        return self.df.copy()

    def get_drop_candidates(self):
        return sorted(self.drop_candidates)


class FeatureEngineering:
    """Create derived features, transform values, and return a processed copy."""

    def __init__(self, df):
        self.df = df.copy()
        self.scaler = None

    def basic_info(self):
        print(f"Rows: {self.df.shape[0]:,}\nColumns: {self.df.shape[1]:,}")
        print(self.df.columns.tolist())

    def drop_columns(self, columns):
        existing = [column for column in columns if column in self.df]
        missing = [column for column in columns if column not in self.df]
        if existing:
            self.df = self.df.drop(columns=existing)
            print(f"Removed: {existing}")
        if missing:
            print(f"Columns not found: {missing}")

    def create_time_features(self):
        if "valid_time" not in self.df:
            print("'valid_time' not found.")
            return
        self.df["hour"] = self.df["valid_time"].dt.hour
        self.df["day"] = self.df["valid_time"].dt.day
        self.df["month"] = self.df["valid_time"].dt.month
        self.df["day_of_week"] = self.df["valid_time"].dt.dayofweek
        self.df["day_of_year"] = self.df["valid_time"].dt.dayofyear

    def create_wind_speed(self):
        u, v = "10m_u_component_of_wind", "10m_v_component_of_wind"
        if u not in self.df or v not in self.df:
            print("Wind component columns not found.")
            return
        self.df["wind_speed"] = np.sqrt(self.df[u] ** 2 + self.df[v] ** 2)

    def create_dewpoint_depression(self):
        temperature, dewpoint = "2_m_air_temperature", "2m_dewpoint_temperature"
        if temperature not in self.df or dewpoint not in self.df:
            print("Temperature or dewpoint column not found.")
            return
        self.df["dewpoint_depression"] = self.df[temperature] - self.df[dewpoint]

    def detect_outliers(self, columns=None):
        columns = columns or self.df.select_dtypes(include=np.number).columns.tolist()
        results = {}
        for column in columns:
            if column not in self.df:
                continue
            series = self.df[column].dropna()
            if series.empty:
                continue
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers = (self.df[column] < lower) | (self.df[column] > upper)
            results[column] = {"Q1": q1, "Q3": q3, "IQR": iqr, "Lower_Bound": lower, "Upper_Bound": upper, "Outliers": outliers.sum(), "Percentage": round(outliers.mean() * 100, 2)}
        return pd.DataFrame(results).T

    def remove_outliers(self, columns=None):
        columns = columns or self.df.select_dtypes(include=np.number).columns.tolist()
        mask = pd.Series(True, index=self.df.index)
        for column in columns:
            if column not in self.df:
                continue
            q1, q3 = self.df[column].quantile([0.25, 0.75])
            iqr = q3 - q1
            mask &= self.df[column].isna() | self.df[column].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        self.df = self.df.loc[mask].reset_index(drop=True)

    def fit_standard_scaler(self, columns):
        columns = [column for column in columns if column in self.df]
        if columns:
            self.scaler = StandardScaler()
            self.df[columns] = self.scaler.fit_transform(self.df[columns])

    def fit_minmax_scaler(self, columns):
        columns = [column for column in columns if column in self.df]
        if columns:
            self.scaler = MinMaxScaler()
            self.df[columns] = self.scaler.fit_transform(self.df[columns])

    def log_transform(self, columns):
        for column in columns:
            if column in self.df and not (self.df[column] < 0).any():
                self.df[column] = np.log1p(self.df[column])

    def check_skewness(self):
        numeric = self.df.select_dtypes(include=np.number)
        return numeric.skew().sort_values(ascending=False)

    def target_correlation(self, target_column):
        if target_column not in self.df:
            print(f"'{target_column}' not found.")
            return None
        return self.df.select_dtypes(include=np.number).corr()[target_column].sort_values(ascending=False)

    def get_data(self):
        return self.df.copy()