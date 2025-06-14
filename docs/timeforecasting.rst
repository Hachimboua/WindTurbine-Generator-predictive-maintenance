Time Series Forecasting Utilities
=================================

Overview
------------------

This module provides utilities for loading, preprocessing, forecasting, and plotting time series data using a BiLSTM model.

Function Reference
------------------

.. autofunction:: load_data

.. autofunction:: preprocess_data

.. autofunction:: load_model

.. autofunction:: predict

.. autofunction:: inverse_transform

.. autofunction:: plot_forecast


Function Details
----------------

.. function:: load_data(filepath: str) -> pandas.DataFrame

   Load time series data from a CSV file.

   :param filepath: Path to the CSV file.
   :return: Loaded data as a pandas DataFrame.

.. function:: preprocess_data(data: pandas.DataFrame, feature_cols: list[str], scaler_path: str) -> numpy.ndarray

   Standardize input data using pre-fitted scaler.

   :param data: Raw input data as a DataFrame.
   :param feature_cols: List of feature column names to include.
   :param scaler_path: Path to the saved scaler (.pkl or .joblib file).
   :return: Scaled feature matrix as a NumPy array.

.. function:: load_model(model_path: str, input_size: int, hidden_size: int, num_layers: int, output_size: int, device: str) -> torch.nn.Module

   Load a trained BiLSTM model for inference.

   :param model_path: Path to the saved model weights.
   :param input_size: Number of input features.
   :param hidden_size: Hidden layer size.
   :param num_layers: Number of LSTM layers.
   :param output_size: Number of output predictions.
   :param device: Device to load model on ('cpu' or 'cuda').
   :return: Loaded PyTorch model.

.. function:: predict(model: torch.nn.Module, input_seq: numpy.ndarray, device: str) -> numpy.ndarray

   Run inference on the input sequence using the BiLSTM model.

   :param model: Trained PyTorch model.
   :param input_seq: Preprocessed and windowed input sequence.
   :param device: Device for inference.
   :return: Model predictions as NumPy array.

.. function:: inverse_transform(predictions: numpy.ndarray, scaler_path: str, target_col_index: int) -> numpy.ndarray

   Invert the scaling transformation on model predictions.

   :param predictions: Scaled predictions.
   :param scaler_path: Path to the saved scaler.
   :param target_col_index: Index of the target variable to inverse-transform.
   :return: Inverse-transformed predictions.

.. function:: plot_forecast(true_values: numpy.ndarray, predicted_values: numpy.ndarray, title: str)

   Plot true vs. predicted values.

   :param true_values: Ground truth RUL values.
   :param predicted_values: Forecasted RUL values.
   :param title: Title for the plot.
