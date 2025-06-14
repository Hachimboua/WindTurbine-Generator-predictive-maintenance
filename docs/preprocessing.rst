Data Preprocessing
==================

Data preprocessing is a crucial step before feeding time-series data into the deep learning models. The dashboard handles basic preprocessing steps internally.

Data Loading (`load_data`)
--------------------------

The `load_data` function in `Dashboard_App/app.py` is responsible for reading your uploaded CSV files into a Pandas DataFrame.

It performs the following:

* Reads the CSV using `pd.read_csv()`.
* **Creates a 'Date' column:** By default, it generates a synthetic 'Date' column based on the DataFrame's numerical index, assuming daily observations starting from '2023-01-01'.

    .. code-block:: python

        df['Date'] = pd.to_datetime('2023-01-01') + pd.to_timedelta(df.index, unit='days')

    If your CSV already contains a specific date/timestamp column, you should modify this line within `load_data` to parse your actual date column (e.g., `df['Date'] = pd.to_datetime(df['Your_Timestamp_Column'])`).

Feature Scaling (`StandardScaler`)
----------------------------------

Before model training or inference, all numerical features in your time-series data are scaled using `sklearn.preprocessing.StandardScaler`. This is performed in the "Train New Model" page before creating sequences, and implicitly by the `run_direct_forecast` function when loading the pre-trained `main_scaler.joblib`.

The `StandardScaler` transforms data such that it has a mean of 0 and a standard deviation of 1. This normalization is essential for deep learning models, as it helps:

* Prevent features with larger numerical ranges from dominating the learning process.
* Speed up convergence during training.

The scaling process involves:

1.  **Fitting:** The `StandardScaler` learns the mean and standard deviation of each feature from the training data (or `main_scaler.joblib` for forecasting).
2.  **Transforming:** This learned scaling is then applied to both the training data and any new data (for forecasting) to ensure consistency.

The columns used for scaling are all numerical columns present in your uploaded CSV, excluding the 'Date' column which is only for plotting.