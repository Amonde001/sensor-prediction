# Climate and Soil Moisture Forecasting Project

## Purpose and Executive Summary

This document presents the end-to-end design of the climate and road pavement soil moisture forecasting project. It explains how the source records are prepared, how the soil moisture and climate models are trained, and how historical and future predictions are produced.

Two modelling systems are used:

1. A multi-output soil moisture model predicts `Port 1` through `Port 5` from climate and time-derived features.
2. A multi-output climate model predicts climate variables using recent climate observations, calendar variables, location information, and an interval of two hours by default.

The five moisture Port values are not used as targets or inputs by the climate forecasting system. They are also documented separately from the climate model workflow.

## Environment Configurations

### Technical environment

The project was developed for Windows using Python 3.12.8. The validated packages include pandas 2.3.3 and scikit-learn 1.6.1. Core modelling and reporting packages are pandas, NumPy, scikit-learn, joblib, matplotlib, seaborn, statsmodels, Jupyter, and nbconvert.

The climate scripts also rely on the following packages when specific notebook operations are used: `tqdm`, `colorama`, `openpyxl`, and `nbformat`.

### Python environment setup

```{r python-setup, eval=FALSE}
# Create and activate a Python environment from the terminal
python -m venv .venv
.venv\\Scripts\\activate

# Install the main scientific and notebook packages
python -m pip install pandas numpy scikit-learn joblib matplotlib seaborn statsmodels jupyter openpyxl tqdm colorama nbformat
```

### R and R Markdown setup

This overview can be rendered in RStudio or by the R Markdown command line. If the document only presents the project and links to existing outputs, R packages are not required for the analytical calculations. The following packages support an HTML report and embedded file navigation:

```{r r-setup, eval=FALSE}
install.packages(c("rmarkdown", "knitr", "htmltools"))

rmarkdown::render("project_executive_overview.Rmd")
```

### Reproducibility conventions

Model training uses fixed random seeds where random procedures are available. Time-based records are sorted before feature construction and model evaluation. All evaluation splits respect chronology, preventing future observations from leaking into training.

The project defines weather observations at two-hour intervals. The climate forecast interval can be changed when calling the forecast function, with the default remaining two hours.

## Data Loading Process for Climate Data

The climate pipeline loads ERA5 CSV sources containing temperature and dew point, pressure and precipitation, radiation, surface temperature, snow information, soil temperature, soil water, and wind components.

`LoadData` in [dataset_processor.py](dataset_processor.py) discovers all CSV files in the climate source directory and presents the files under named datasets. `Processor` then standardizes the timestamp column, rounds location coordinates, and performs inner joins across climate sources using:

```text
valid_time, latitude, longitude
```

A common issue in climate tabular systems is inconsistent variable names and timestamp formats. The processing class therefore converts timestamps to UTC, floors them to hourly precision, merges the climate sources on time and coordinates, and renames the short ERA5 codes to descriptive names.

The ingestion class in [loader.py](loader.py) provides a reusable alternative for loading both CSV and Excel sources, validating arguments, reporting load failures, processing date columns, and combining successfully loaded tables.

## Data Loading Process for Soil Moisture Data

The soil moisture source is an Excel workbook. It contains timestamped pavement sensor readings for `Port 1`, `Port 2`, `Port 3`, `Port 4`, and `Port 5`.

The loading workflow in [main.py](main.py) reads the moisture workbook from its processed data sheet, assigns stable column names, skips the workbook metadata rows, and converts the timestamp column for merging.

The intended moisture schema is:

| Field | Purpose |
|---|---|
| `valid_time` | Chronological merge key |
| `Port 1` | Pavement moisture target 1 |
| `Port 2` | Pavement moisture target 2 |
| `Port 3` | Pavement moisture target 3 |
| `Port 4` | Pavement moisture target 4 |
| `Port 5` | Pavement moisture target 5 |

The soil moisture loading and processing behaviour is also demonstrated in [moisture_data.py](moisture_data.py) and [road_soil_moisture_prediction.ipynb](road_soil_moisture_prediction.ipynb).

## Data Merging

The climate and moisture systems are joined on `valid_time`. The merger in [merger.py](merger.py) validates the merge column, accepted merge type, suffixes, and DataFrame structure before producing the final table.

The normal workflow uses an inner join because both climate and pavement observations are required for supervised soil moisture training. The resulting merged observation contains climate inputs and the five moisture targets at the same timestamp.

The main pipeline proceeds in this order:

1. Load and validate the climate files.
2. Load and validate the soil moisture workbook.
3. Standardize and parse timestamps.
4. Inner-join both sources on `valid_time`.
5. Create calendar, interaction, lag, and rolling features.
6. Remove time periods and incomplete records that are unsuitable for leakage-safe evaluation.

The pipeline implementation is orchestrated in [main.py](main.py), with the detailed feature engineering and analysis documented in [climate_moisture_feature_pipeline.ipynb](climate_moisture_feature_pipeline.ipynb).

## Soil Moisture Model

### Prediction objective

The soil moisture model predicts five pavement sensor readings simultaneously. It treats `Port 1` through `Port 5` as a multi-output target and uses climate observations plus derived calendar and historical features as predictors.

### How the model is built

The soil moisture notebook compares several regression approaches and feature configurations. Candidate models include:

- Dummy baseline regression for reference
- Linear Regression
- Ridge Regression
- Random Forest Regressor
- Extra Trees Regressor
- Hist Gradient Boosting Regressor where used during the comparison

Feature preparation includes:

- Raw climate variables
- Calendar variables such as hour, weekday, month, and day of year
- Cyclic calendar encodings for repeating daily and yearly patterns
- Soil temperature and soil water aggregate features
- Wind speed derived from wind components
- Three-step historical lag values for climate variables
- Three-step and six-step rolling averages
- Historical lag values for each moisture Port

The final saved soil moisture model is described as a multi-output `LinearRegression` model in the model metadata. It was selected after comparison and fitted with the available feature configuration. Linear regression provides a coefficient for each input-to-output relationship and creates predictions as weighted sums of the final standardized feature matrix.

### How evaluation works

The soil moisture workflow uses chronological partitions rather than a random split:

| Partition | Expected purpose |
|---|---|
| Training | Fit model coefficients and learn feature relationships |
| Validation | Compare feature configurations and models |
| Test | Provide a final untouched performance estimate |

This design preserves time order and reduces the risk of a model learning from observations that occur later than its training period. Reported measures include:

- R, or Pearson correlation
- R-squared
- Mean Absolute Error
- Mean Squared Error
- Root Mean Squared Error
- Median Absolute Error
- Maximum Absolute Error
- Explained variance

## Soil Moisture Prediction for Past Data

Historical soil moisture predictions are generated for observations represented in the merged and engineered data. For each row, the model receives climate and calendar inputs available to the prediction process and returns predictions for all five Ports.

The resulting historical output is stored with actual and predicted columns for each sensor. The deliverable supports direct comparisons across timestamps and makes it possible to calculate residuals:

```text
Residual = Actual Port Value - Predicted Port Value
```

The historical result is available in [data/processed/soil_moisture_predictions.csv](data/processed/soil_moisture_predictions.csv). The complete modelling procedure and validation analysis are shown in [road_soil_moisture_prediction.ipynb](road_soil_moisture_prediction.ipynb).

## Soil Moisture Prediction for Future Data

Future soil moisture forecasting uses the designed prediction horizon and the model's configured feature availability. The forecasting implementation must avoid using values that would not yet be known at the time a forecast is issued.

When recursive forecasting is used, a predicted sensor or climate state may become one of the later lagged inputs. The quality of long-range predictions should therefore be interpreted carefully because small errors can accumulate over successive forecast steps.

Future soil moisture forecasts are intended for decision support and operational planning. They should not be treated as measurements, and uncertainty should be reassessed when new observations become available.

## Climate Forecasting Model

### Prediction objective

The climate forecasting model is a reusable multi-output regressor for numeric climate variables. It explicitly excludes the five pavement Port readings from model targets and inputs.

The implementation is in [climate_forecasting.py](climate_forecasting.py).

### How the climate model is built

The climate workflow sorts observations chronologically, checks duplicate timestamps, and identifies numeric variables as climate targets. Latitude and longitude are used as location context. For the 2-hour interval, the model performs time-based interpolation onto a regular grid.

Each training row contains:

- Three lagged values for every climate target
- Latitude and longitude
- Hour of day
- Day of week
- Calendar month
- Day of year
- Cyclic hour encoding
- Cyclic day-of-year encoding

The selected climate estimator is an `ExtraTreesRegressor` containing 180 trees. Extra Trees combines many randomized decision trees and averages the predictions from all trees for each target. It is suitable for nonlinear climate relationships and produces simultaneous predictions across multiple climate outputs.

The training design uses the earlier 80 percent of observations for fitting and the latest 20 percent for a chronological holdout test. No `Port` column is introduced into this process.

### Climate performance interpretation

The climate notebook reports R, R-squared, MAD, MAE, MSE, RMSE, median absolute error, and maximum absolute error by target and as a flattened overall result.

The overall chronological holdout result provides a compact indication of accuracy, but Directors should review the per-variable values because climate features use different physical units and scales.

## Past Climate Predictions

The function named `predict_entire_raw_dataset()` attaches actual and predicted columns to the complete raw observation table. For every climate target, it creates:

```text
{climate_variable}_actual
{climate_variable}_predicted
```

Historical prediction generation uses actual lagged climate observations, which makes the past prediction workflow distinct from recursive future forecasting.

The resulting actual-versus-predicted climate output is available as [data/processed/climate_predictions_with_actual.csv](data/processed/climate_predictions_with_actual.csv). The visual and executable explanation is available in [climate_forecasting_notebook.ipynb](climate_forecasting_notebook.ipynb).

## Future Climate Predictions

The climate forecasting module exposes a function with this professional calling pattern:

```python
from climate_forecasting import climate_forecast

forecast = climate_forecast(
    years=1,
    interval_hours=2.0,
    output_path="climate_forecast.csv"
)
```

The function calculates the requested forecast horizon from a selected number of years and interval in hours. It starts after the latest observed timestamp and creates timestamps at regular interval increments.

For each future step, the model:

1. Builds the previous three climate values as lag features.
2. Adds calendar and location features for the new timestamp.
3. Predicts the complete climate vector.
4. Appends the prediction to the rolling climate history.
5. Uses the new climate vector to prepare the next future timestamp.

No soil moisture Port column is included in future climate forecasting.

An example generated forecast is available in [climate_forecast.csv](climate_forecast.csv).

## Key Code and Report Links

This section provides navigational links to the project resources that support the two modelling systems. Source datasets and model artifact files are intentionally not linked here.

### Main workflow and reusable modules

- [main.py](main.py): Pipeline orchestration
- [loader.py](loader.py): CSV and Excel loading utility
- [merger.py](merger.py): Validated multi-source merger
- [dataset_processor.py](dataset_processor.py): Climate loading, standardization, cleaning, and feature support
- [moisture_data.py](moisture_data.py): Soil moisture data processing
- [climate_forecasting.py](climate_forecasting.py): Climate model, historical prediction function, and future forecast function

### Analytical notebooks and report

- [road_soil_moisture_prediction.ipynb](road_soil_moisture_prediction.ipynb): Soil moisture model development, comparison, evaluation, and future prediction procedure
- [climate_moisture_feature_pipeline.ipynb](climate_moisture_feature_pipeline.ipynb): Climate and moisture feature preparation pipeline
- [climate_forecasting_notebook.ipynb](climate_forecasting_notebook.ipynb): Climate model training, metrics, past predictions, and future forecasts
- [climate_soil_moisture_prediction_report.docx](climate_soil_moisture_prediction_report.docx): Project report document

### Prediction outputs

- [data/processed/soil_moisture_predictions.csv](data/processed/soil_moisture_predictions.csv): Historical soil moisture actuals and predictions
- [data/processed/climate_predictions_with_actual.csv](data/processed/climate_predictions_with_actual.csv): Historical climate actuals and predictions
- [climate_forecast.csv](climate_forecast.csv): Example future climate forecast

## Executive Conclusions

The project combines weather intelligence with pavement moisture monitoring in a controlled, repeatable modelling workflow. The soil moisture model uses multi-output supervised regression across five sensor targets. The climate model uses lagged observations and tree-based nonlinear regression to forecast all climate variables at configurable future intervals.

The strongest controls for trustworthy use are chronology-aware evaluation, explicit exclusion of future information, clear separation of actual and predicted values, and the exclusion of moisture Port columns from the climate model.

Long-range recursive predictions naturally carry more uncertainty than short-range predictions. For operational decisions, model output should be reviewed alongside new measurements, physical sensor checks, and domain-specific thresholds.
