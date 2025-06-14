Data Requirements
=================

For both forecasting and training, your CSV data should meet the following general requirements:

* **Time-Series Format:** Each row should represent a sequential observation in time.
* **Numerical Features:** All columns intended for input to the model (features) must contain numerical data. Non-numerical columns will be ignored or might cause issues.
* **Target Columns:** The CSV must include the following columns for the model's output targets. These are the values the model is trained to predict:
    * `degradation`
    * `temperature_estimated`
    * `time_since_last_maintenance`
* **Date Column (Recommended):**
    While the app can generate a synthetic `Date` column if none exists, it's highly recommended that your CSV includes a proper date/timestamp column. This improves the interpretability of plots.
    * If your CSV already has a specific date/time column (e.g., 'Timestamp', 'ReadingDate', 'EventTime'), you **must** modify the `load_data` function in `Dashboard_App/app.py` to correctly parse and use your specific date column. For example, if your column is named `timestamp`:

        .. code-block:: python

            @st.cache_data
            def load_data(uploaded_file):
                df = pd.read_csv(uploaded_file)
                df['Date'] = pd.to_datetime(df['timestamp']) # Use your actual column name
                return df

    * Ensure the frequency of your data matches the `freq` parameter used in `pd.date_range` within `run_direct_forecast` (e.g., `'D'` for daily, `'H'` for hourly).
* **Sufficient Data:**
    * For **forecasting**, your dataset needs at least `50` rows (the default `window_size`) to create the initial input sequence for prediction.
    * For **training**, a larger dataset is generally required for effective model learning and generalization. The minimum length for `create_direct_sequences` is `window_size + future_steps - 1`.