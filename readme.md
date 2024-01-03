# Time Series Energy Forecasting using SARIMA
This project applies the Seasonal AutoRegressive Integrated Moving Average (SARIMA) model to forecast energy consumption. The analysis is performed on a dataset containing historical energy consumption data, aiming to predict future values with a focus on capturing the seasonality in the data.
## Project Structure

TIME-SERIES-ENERGY-FORECASTING-USING-SARIMA/
│
├── .venv/
│
├── data/
│   └── energy_consumption_2.csv
├── notebook/
│   └── time_series_energy_forecast_sarima.ipynb
│
├── scripts/
│   └── time_series_energy_forecasting_sarima.py
├── .gitignore
├── readme.md
└── requirements.txt
```

## Requirements

To run the project, you need the following libraries:

- pandas
- numpy
- matplotlib
- scikit-learn
- statsmodels

You can install all the necessary packages using the `requirements.txt` file by running:

```shell
pip install -r requirements.txt
```

## Dataset

The dataset `energy_consumption_2.csv` includes the following columns:

- `DATE`: The date of the record.
- `ENERGY_INDEX`: The energy consumption index.

## Usage

1. Load and preprocess the data, including conversion of the date column to datetime objects.
2. Split the dataset into training and test sets.
3. Conduct stationarity tests using Dickey-Fuller and KPSS tests.
4. Transform the series to achieve stationarity if necessary.
5. Generate Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots.
6. Fit the SARIMA model to the dataset.
7. Forecast the energy consumption for the next 36 months.
8. Visualize the original data along with the forecasted values.

## Running the Analysis

To execute the time series forecasting analysis, navigate to the `scripts` directory and run the Python script:

```shell
python time_series_energy_forecasting_sarima.py
```

Alternatively, you can run the Jupyter notebook `time_series_energy_forecast_sarima.ipynb` for an interactive experience.

## Outputs

The script and notebook will output several plots showing:

- The original training and validation data.
- The stationarity of the series after differencing.
- The ACF and PACF plots.
- The forecast on the test dataset.
- The forecast for future dates beyond the dataset.

## Contributing

Contributions are welcome. Please open an issue or submit a pull request with your suggestions.

## License

[MIT](LICENSE)

