Introduction
============

.. include:: ../README.md
   :parser: myst_parser.sphinx_

Overview
--------

The **Predictive Maintenance AI Dashboard** is a powerful Streamlit application designed to help predict the health and Remaining Useful Life (RUL) of industrial components. Leveraging advanced deep learning models, it allows users to forecast key health indicators and determine when a component is likely to fail, enabling proactive maintenance strategies.

This dashboard offers a user-friendly interface for both running pre-trained models and training new models on custom time-series data.

Features
--------

* **Live RUL Prediction:** Forecast a component's degradation and estimate its Remaining Useful Life (RUL) in operating cycles.
* **Component Health Forecasting:** Visualize future trajectories for critical health indicators like degradation, temperature, and time since last maintenance.
* **Flexible Model Selection:** Choose between various deep learning architectures (BiLSTM, BiGRU, Conv1D-LSTM) for forecasting.
* **Custom Model Training:** Upload your own time-series data to train new predictive models from scratch.
* **Hyperparameter Configuration:** Customize training parameters such as window size, future steps, epochs, and learning rate.
* **Model Persistence:** Download newly trained models (`.pth`) and their associated scalers (`.joblib`) for local storage and future use.
* **Interactive Visualizations:** All forecasts and training progress are displayed using interactive Plotly charts, allowing for detailed inspection.