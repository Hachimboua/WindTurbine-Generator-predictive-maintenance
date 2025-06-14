Troubleshooting
===============

This section addresses common issues you might encounter while using or developing the Predictive Maintenance AI Dashboard.

`TypeError: Addition/subtraction of integers and integer-arrays with Timestamp is no longer supported...`
------------------------------------------------------------------------------------------------------

**Problem:** This error arises from an incompatibility between Plotly's internal handling of datetime objects and certain Pandas versions when using line/shape annotations (`add_vline`, `add_shape`). Plotly's internal functions might attempt to perform arithmetic operations on datetime strings or badly-inferred datetime objects, which Pandas' `Timestamp` objects no longer directly support with integer operations.

**Solution:** The current code explicitly converts the `pandas.Timestamp` object representing the `prediction_point_date` to an ISO-formatted string (`.strftime('%Y-%m-%d %H:%M:%S')`) when passed to `fig.add_shape`. This is the most reliable way to provide date values to Plotly shapes/annotations to avoid type conflicts.

* **Check Plotly and Pandas Versions:** Ensure your `plotly` and `pandas` libraries are reasonably up-to-date. Sometimes, updating these libraries can resolve such inter-library compatibility issues.
* **Verify `strftime` Format:** While the current format `'%Y-%m-%d %H:%M:%S'` is generally robust, ensure it matches the precision or format Plotly expects if issues persist.

"Model/scaler not found" Error
------------------------------

**Problem:** When you select a pre-trained model on the "Forecasting" page, the application reports that the model (`.pth`) or scaler (`.joblib`) file is missing.

**Solution:**
* **Verify File Paths:** Ensure that the pre-trained model files (e.g., `model_BiLSTM.pth`, `model_BiGRU.pth`, `model_Conv1D_LSTM.pth`) and the `main_scaler.joblib` file are located in the `Dashboard_App` directory within your project structure.
* **Train a New Model:** If you don't have these files, you can train a new model using the "Train New Model" page in the dashboard. After training, remember to click the "Download Model" and "Download Scaler" buttons, and then manually place these downloaded files into the `Dashboard_App` folder.

"Uploaded data is not long enough"
----------------------------------

**Problem:** When uploading a CSV file, especially on the "Forecasting" page, you receive an error indicating that the data is too short.

**Solution:**
* The application requires a minimum `window_size` (default 50 rows) of historical data to create the input sequence for the models.
* Ensure your uploaded CSV file has at least `50` rows for the forecasting and training features to function correctly. For better training results, much larger datasets are recommended.

Missing Plots / Blank Screen
----------------------------

**Problem:** The Streamlit app runs, but the Plotly charts do not display, showing a blank space or an error message in the browser's console.

**Solution:**
* **Check Browser Developer Console:** Open your browser's developer tools (usually F12) and check the "Console" tab for any JavaScript errors. These errors often provide clues about why Plotly charts might not be rendering.
* **Verify `plotly` Installation:** Ensure `plotly` is correctly installed in your virtual environment: `pip show plotly`. If it's not listed, reinstall it: `pip install plotly`.
* **Streamlit Version Compatibility:** While less common, very old or very new Streamlit versions might have minor rendering quirks with Plotly. Ensure your Streamlit version is reasonably up-to-date.
* **Data Issues:** Occasionally, malformed data or `NaN` values in the data being plotted can cause Plotly to fail. Check `df_history.head()` and `forecast_df.head()` to ensure data integrity.