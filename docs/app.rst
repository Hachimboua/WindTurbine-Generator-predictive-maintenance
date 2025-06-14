Direct Forecast Dashboard Documentation
======================================

Key Features
------------

Code Structure
--------------

Error Handling
--------------
.. toctree::
   :maxdepth: 2
   :caption: Related Documentation:
   
   installation.rst
   usage.rst
   preprocessing.rst
   modelstesting.rst
   timeforecasting.rst
   
   modules.rst

Overview
--------

This Streamlit-based dashboard provides predictive maintenance capabilities using a Bidirectional LSTM (BiLSTM) deep learning model. It forecasts:

* Remaining Useful Life (RUL) of a component by predicting when degradation will cross a failure threshold.
* Multi-variable forecasts (degradation, temperature, maintenance time) for component health monitoring.

For installation instructions, see :doc:`installation`. For usage examples, refer to :doc:`usage`.

Key Features
-----------

* **Two Modes**:
    * Live RUL Prediction (see :doc:`timeforecasting`)
    * Component Health Forecasting
* **Interactive Visualization**: Uses Plotly for dynamic, zoomable charts.
* **Scalable Input Handling**: Accepts CSV files with historical sensor data (details in :doc:`preprocessing`).
* **Caching for Performance**: Optimizes model loading and data processing.

Code Structure
-------------

1. **App Configuration** (``st.set_page_config``)
    * Sets up the dashboard layout with a title ("Direct Forecast Dashboard") and an icon (⚡).

2. **Model & Configuration** 
    * See model architecture details in :doc:`modules`
    * Testing results available in :doc:`modetesting1`

3. **Cached Helper Functions**
    * Implementation details in :doc:`utils`

5. **Streamlit UI**
    * **Sidebar Controls**
        * Page Selection: Toggles between RUL Prediction and Component Health Forecasting.
        * File Uploader: Accepts CSV files with historical data.
        * Forecast Starting Point: Slider to select where the forecast begins.

How to Use
----------

See comprehensive usage guide in :doc:`usage`.

Dependencies
------------

* **Python Libraries**:
    * Full list in :doc:`requirements.txt <requirements>`
* **Model Files**:
    * ``model_BiLSTM.pth`` (trained PyTorch model)
    * ``main_scaler.joblib`` (feature scaler)

Error Handling
-------------

* For troubleshooting, see :doc:`usage` and :doc:`utils`.