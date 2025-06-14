Time Series Forecasting Logic
=============================

This section describes the core logic behind generating forecasts using the trained models.

The `run_direct_forecast` Function
----------------------------------

The `run_direct_forecast` function is responsible for taking a historical window of data, feeding it through the loaded model, and producing future predictions.

**Inputs:**

* **`model`**: The loaded PyTorch deep learning model (e.g., BiLSTM, BiGRU).
* **`scaler`**: The fitted `StandardScaler` used to normalize the input data and inverse-transform the predictions.
* **`history_window_df`**: A Pandas DataFrame containing the recent historical data points (features) that will be used as input to the model. Its length is determined by `window_size`.
* **`forecast_start_date`**: The `pandas.Timestamp` object representing the date immediately preceding the start of the forecast period.

**Process:**

1.  **Extract Numerical History:** Only numerical columns from `history_window_df` are selected, as the 'Date' column is not a feature for the model.
2.  **Align Features:** The historical numerical data is re-indexed to ensure its columns match the `feature_names_in_` of the `scaler`. This is crucial for consistent input to the model. Missing columns are filled with 0.
3.  **Scale Input:** The historical data is transformed using the `scaler` to match the scale the model was trained on.
4.  **Prepare Input Tensor:** The scaled NumPy array is converted into a PyTorch tensor and reshaped (`unsqueeze(0)`) to add a batch dimension, making it suitable for the model's input layer.
5.  **Model Prediction:** The input tensor is passed through the `model` in evaluation mode (`torch.no_grad()`) to obtain a flattened prediction of future values.
6.  **Reshape Prediction:** The flattened prediction is reshaped back into a 2D array, where each row corresponds to a future time step and columns correspond to the target variables (`degradation`, `temperature_estimated`, `time_since_last_maintenance`).
7.  **Inverse Transform Prediction:** The scaled predictions are inverse-transformed using the `scaler` to return them to their original, interpretable scales.
8.  **Create Forecast DataFrame:** A new Pandas DataFrame (`forecast_df`) is created from the inverse-transformed predictions, with columns assigned to the target variables.
9.  **Generate Future Dates:** A 'Date' column is added to `forecast_df`, generating a series of future dates starting from `forecast_start_date` based on the assumed daily frequency of the data.

**Output:**

* A Pandas DataFrame (`forecast_df`) containing the unscaled, forecasted values for `degradation`, `temperature_estimated`, and `time_since_last_maintenance`, along with their corresponding future `Date` values.

Remaining Useful Life (RUL) Calculation
--------------------------------------

After the `degradation` forecast is obtained, the Remaining Useful Life (RUL) is calculated. This is defined as the number of future cycles (or days in our current date setup) until the predicted degradation value reaches or exceeds a predefined failure threshold.

* **Threshold:** The failure threshold for degradation is set at **0.66**.
* **Calculation:** `np.where(forecasted_degradation >= 0.66)[0][0]` finds the index of the first point in the `degradation` forecast where the value is greater than or equal to 0.66. This index directly corresponds to the RUL in cycles (or days).
* **Interpretation:** If the threshold is not met within the entire forecast window (1200 steps), the RUL is reported as `> 1200 cycles`.