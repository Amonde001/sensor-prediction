"""Climate forecasting utilities and reusable model for the moisture-sensor project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)

TIME_COLUMN = "valid_time"
PORT_COLUMNS = {f"Port {i}" for i in range(1, 6)}
METADATA_COLUMNS = {TIME_COLUMN, "latitude", "longitude"}


def identify_climate_targets(frame: pd.DataFrame) -> list[str]:
    """Return numeric climate columns, excluding time, location, and all Port sensors."""
    return [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column not in METADATA_COLUMNS and column not in PORT_COLUMNS
    ]


def load_raw_dataset(data_path: str | Path) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_csv(data_path)
    frame[TIME_COLUMN] = pd.to_datetime(frame[TIME_COLUMN], errors="raise")
    frame = frame.sort_values(TIME_COLUMN).reset_index(drop=True)
    targets = identify_climate_targets(frame)
    if not targets:
        raise ValueError("No climate columns were found in the dataset.")
    if frame[TIME_COLUMN].duplicated().any():
        raise ValueError("The dataset contains duplicate valid_time rows.")
    frame[targets] = frame[targets].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=targets)
    return frame.reset_index(drop=True), targets


class ClimateForecastModel:
    """Multi-output climate forecaster trained from chronological climate observations."""

    def __init__(
        self,
        data_path: str | Path,
        interval_hours: float = 2.0,
        lag_intervals: int = 3,
        random_state: int = 42,
    ) -> None:
        self.data_path = Path(data_path)
        self.interval_hours = float(interval_hours)
        self.lag_intervals = int(lag_intervals)
        self.random_state = random_state
        self.model: ExtraTreesRegressor | None = None
        self.feature_names: list[str] = []
        self.target_names: list[str] = []
        self.metrics_: pd.DataFrame | None = None
        self.train_rows_: int = 0
        self.test_rows_: int = 0

    def _interval_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        numeric = frame.select_dtypes(include=[np.number]).columns.tolist()
        indexed = frame.set_index(TIME_COLUMN)[numeric]
        start = indexed.index.min().ceil(f"{self.interval_hours:g}h")
        end = indexed.index.max().floor(f"{self.interval_hours:g}h")
        grid = pd.date_range(start=start, end=end, freq=f"{self.interval_hours:g}h")
        resampled = indexed.reindex(indexed.index.union(grid)).sort_index().interpolate(method="time")
        resampled = resampled.reindex(grid)
        result = resampled.reset_index().rename(columns={"index": TIME_COLUMN})
        return result.dropna()

    def _calendar_features(self, timestamp: pd.Timestamp, row_features: dict[str, float]) -> None:
        row_features.update({
            "calendar_hour": float(timestamp.hour),
            "calendar_dayofweek": float(timestamp.dayofweek),
            "calendar_month": float(timestamp.month),
            "calendar_dayofyear": float(timestamp.dayofyear),
            "calendar_hour_sin": float(np.sin(2 * np.pi * timestamp.hour / 24)),
            "calendar_hour_cos": float(np.cos(2 * np.pi * timestamp.hour / 24)),
            "calendar_doy_sin": float(np.sin(2 * np.pi * timestamp.dayofyear / 365.25)),
            "calendar_doy_cos": float(np.cos(2 * np.pi * timestamp.dayofyear / 365.25)),
        })

    def build_features(self, frame: pd.DataFrame, targets: Iterable[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        frame = frame.sort_values(TIME_COLUMN).reset_index(drop=True).copy()
        targets = list(targets)
        records: list[dict[str, float]] = []
        labels: list[list[float]] = []

        for position in range(self.lag_intervals, len(frame)):
            timestamp = frame.loc[position, TIME_COLUMN]
            record: dict[str, float] = {}
            for target in targets:
                for lag in range(1, self.lag_intervals + 1):
                    record[f"{target}__lag_{lag}"] = float(frame.loc[position - lag, target])
            for column in ("latitude", "longitude"):
                if column in frame.columns:
                    record[column] = float(frame.loc[position, column])
            self._calendar_features(timestamp, record)
            records.append(record)
            labels.append([float(frame.loc[position, target]) for target in targets])

        features = pd.DataFrame(records)
        labels_frame = pd.DataFrame(labels, columns=targets)
        if len(features) == 0:
            raise ValueError("Not enough rows to build lagged features.")
        return features, labels_frame

    def fit(self) -> "ClimateForecastModel":
        raw, self.target_names = load_raw_dataset(self.data_path)
        interval_frame = self._interval_frame(raw)
        features, labels = self.build_features(interval_frame, self.target_names)

        split = int(len(features) * 0.80)
        split = max(self.lag_intervals, min(split, len(features) - 1))
        x_train, x_test = features.iloc[:split], features.iloc[split:]
        y_train, y_test = labels.iloc[:split], labels.iloc[split:]

        self.feature_names = features.columns.tolist()
        self.model = ExtraTreesRegressor(
            n_estimators=180,
            random_state=self.random_state,
            n_jobs=-1,
            min_samples_leaf=2,
            max_features=1.0,
        )
        self.model.fit(x_train, y_train)

        predictions = self.model.predict(x_test)
        self.train_rows_ = len(x_train)
        self.test_rows_ = len(x_test)
        self.metrics_ = self._metric_table(y_test.to_numpy(), predictions, y_test.columns.tolist())
        return self

    @staticmethod
    def _metric_table(actual: np.ndarray, predicted: np.ndarray, names: list[str]) -> pd.DataFrame:
        rows: list[dict[str, float | str | int]] = []

        for index, name in enumerate(names):
            y_true = actual[:, index]
            y_pred = predicted[:, index]
            r_value = float(np.corrcoef(y_true, y_pred)[0, 1]) if np.std(y_true) > 0 and np.std(y_pred) > 0 else np.nan
            errors = y_true - y_pred
            rows.append({
                "target": name,
                "R": r_value,
                "R2": float(r2_score(y_true, y_pred)),
                "MAD": float(np.median(np.abs(errors - np.median(errors)))),
                "MAE": float(mean_absolute_error(y_true, y_pred)),
                "MSE": float(mean_squared_error(y_true, y_pred)),
                "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "MedAE": float(median_absolute_error(y_true, y_pred)),
                "MaxError": float(np.max(np.abs(errors))),
            })

        y_true_flat = actual.ravel()
        y_pred_flat = predicted.ravel()
        r_flat = float(np.corrcoef(y_true_flat, y_pred_flat)[0, 1]) if np.std(y_pred_flat) > 0 else np.nan
        errors_flat = y_true_flat - y_pred_flat
        rows.append({
            "target": "OVERALL_FLATTENED",
            "R": r_flat,
            "R2": float(r2_score(y_true_flat, y_pred_flat)),
            "MAD": float(np.median(np.abs(errors_flat - np.median(errors_flat)))),
            "MAE": float(mean_absolute_error(y_true_flat, y_pred_flat)),
            "MSE": float(mean_squared_error(y_true_flat, y_pred_flat)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))),
            "MedAE": float(median_absolute_error(y_true_flat, y_pred_flat)),
            "MaxError": float(np.max(np.abs(errors_flat))),
        })
        return pd.DataFrame(rows)

    def predict_raw_dataset(self, data_path: str | Path | None = None, output_path: str | Path | None = None) -> pd.DataFrame:
        """Create actual and predicted columns for every eligible row in the raw dataset."""
        if self.model is None:
            raise RuntimeError("Fit the model before making predictions.")
        path = Path(data_path) if data_path else self.data_path
        raw, targets = load_raw_dataset(path)
        if set(targets) != set(self.target_names):
            raise ValueError("The saved model targets do not match the supplied raw dataset schema.")

        result = raw.drop(columns=[column for column in raw.columns if column in PORT_COLUMNS], errors="ignore").copy()
        predicted: dict[str, np.ndarray] = {target: np.full(len(raw), np.nan) for target in targets}
        prediction_records: list[dict[str, float]] = []
        prediction_positions: list[int] = []
        for position in range(self.lag_intervals, len(raw)):
            record: dict[str, float] = {}
            timestamp = raw.loc[position, TIME_COLUMN]
            for target in targets:
                for lag in range(1, self.lag_intervals + 1):
                    record[f"{target}__lag_{lag}"] = float(raw.loc[position - lag, target])
            record["latitude"] = float(raw.loc[position, "latitude"])
            record["longitude"] = float(raw.loc[position, "longitude"])
            self._calendar_features(timestamp, record)
            prediction_records.append(record)
            prediction_positions.append(position)

        prediction_frame = pd.DataFrame(prediction_records, columns=self.feature_names)
        prediction_values = self.model.predict(prediction_frame)
        for row_index, position in enumerate(prediction_positions):
            for target_index, target in enumerate(targets):
                predicted[target][position] = float(prediction_values[row_index, target_index])

        for target in targets:
            result[f"{target}_actual"] = result[target]
            result[f"{target}_predicted"] = predicted[target]
        if output_path:
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(destination, index=False)
        return result

    def forecast(self, years: float, interval_hours: float = 2.0, output_path: str | Path | None = None) -> pd.DataFrame:
        """Recursive future forecast with no Port sensor inputs or outputs."""
        if self.model is None:
            raise RuntimeError("Fit the model before forecasting.")
        if years <= 0:
            raise ValueError("years must be greater than zero.")
        if interval_hours <= 0:
            raise ValueError("interval_hours must be greater than zero.")

        raw, targets = load_raw_dataset(self.data_path)
        interval_frame = self._interval_frame_for_hours(raw, interval_hours)
        history = interval_frame[self.target_names].copy()
        latitude = float(interval_frame["latitude"].iloc[-1])
        longitude = float(interval_frame["longitude"].iloc[-1])

        last_timestamp = interval_frame[TIME_COLUMN].max()
        horizon_steps = int(np.ceil(years * 365.25 * 24 / interval_hours))
        future_rows: list[dict[str, float | pd.Timestamp]] = []

        for step in range(1, horizon_steps + 1):
            timestamp = last_timestamp + pd.Timedelta(hours=interval_hours * step)
            record: dict[str, float] = {}
            for target in self.target_names:
                for lag in range(1, self.lag_intervals + 1):
                    if len(history) >= lag:
                        record[f"{target}__lag_{lag}"] = float(history[target].iloc[-lag])
                    else:
                        record[f"{target}__lag_{lag}"] = float(interval_frame[target].iloc[0])
            record["latitude"] = latitude
            record["longitude"] = longitude
            self._calendar_features(timestamp, record)
            prediction = self.model.predict(pd.DataFrame([record], columns=self.feature_names))[0]
            row: dict[str, float | pd.Timestamp] = {TIME_COLUMN: timestamp}
            for target, value in zip(self.target_names, prediction):
                row[target] = float(value)
            future_rows.append(row)
            history = pd.concat([history, pd.DataFrame([[float(row[t]) for t in self.target_names]], columns=self.target_names)], ignore_index=True)

        forecast_frame = pd.DataFrame(future_rows)
        forecast_frame.insert(1, "latitude", latitude)
        forecast_frame.insert(2, "longitude", longitude)
        for target in self.target_names:
            forecast_frame[target] = forecast_frame[target].astype(float)
        if output_path:
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            forecast_frame.to_csv(destination, index=False)
        return forecast_frame

    def _interval_frame_for_hours(self, frame: pd.DataFrame, interval_hours: float) -> pd.DataFrame:
        original = self.interval_hours
        self.interval_hours = interval_hours
        try:
            return self._interval_frame(frame)
        finally:
            self.interval_hours = original

    def save(self, model_path: str | Path, metadata_path: str | Path | None = None) -> None:
        destination = Path(model_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination, compress=3)
        if metadata_path:
            metadata = {
                "model_file": str(destination),
                "model_type": "ExtraTreesRegressor multi-output forecaster",
                "interval_hours_default": self.interval_hours,
                "lag_intervals": self.lag_intervals,
                "target_columns": self.target_names,
                "train_rows": self.train_rows_,
                "test_rows": self.test_rows_,
                "port_columns_used_as_targets": False,
                "raw_dataset_function": "predict_raw_dataset",
                "future_forecast_function": "forecast(years, interval_hours, output_path)",
            }
            Path(metadata_path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "climate_model.joblib"
DEFAULT_RAW_PATH = Path(__file__).resolve().parent / "data" / "processed" / "merged_full_data.csv"


def load_climate_model(model_path: str | Path | None = None) -> ClimateForecastModel:
    """Load the saved climate model from disk."""
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    return joblib.load(path)


def predict_entire_raw_dataset(
    data_path: str | Path | None = None,
    output_path: str | Path | None = None,
    model_path: str | Path | None = None,
) -> pd.DataFrame:
    """Attach actual and predicted climate columns to the complete raw dataset."""
    model = load_climate_model(model_path)
    return model.predict_raw_dataset(
        data_path=data_path or DEFAULT_RAW_PATH,
        output_path=output_path,
    )


def climate_forecast(
    years: float,
    interval_hours: float = 2.0,
    output_path: str | Path | None = None,
    model_path: str | Path | None = None,
    data_path: str | Path | None = None,
) -> pd.DataFrame:
    """Return and optionally save future climate predictions for a given horizon and interval."""
    model = load_climate_model(model_path)
    if data_path:
        model.data_path = Path(data_path)
    frame = model.forecast(
        years=years,
        interval_hours=interval_hours,
        output_path=output_path,
    )
    return frame
