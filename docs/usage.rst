Usage Guide
===========

The Predictive Maintenance AI Dashboard provides two primary functionalities:
forecasting with pre-trained models and training new custom models.

Forecasting Component Health and RUL
-----------------------------------

To get started with forecasting:

1.  **Select a Pre-Trained Model:** Choose the model architecture (e.g., BiLSTM, BiGRU) from the sidebar.
2.  **Upload Your Data:** Provide your historical sensor or operational data in CSV format.
3.  **Define Forecast Start:** Specify the point in your historical data from which the prediction should begin.
4.  **Visualize Results:** Review the interactive plots displaying predicted degradation, temperature, and maintenance intervals, along with the derived Remaining Useful Life (RUL).

Detailed steps for the forecasting interface are available in :doc:`app`.

Training a New Model
--------------------

If you wish to build a custom model tailored to your specific data:

1.  **Upload Training Data:** Supply your dataset in CSV format.
2.  **Configure Training Parameters:** Adjust hyperparameters such as window size, future steps, epochs, and learning rate.
3.  **Start Training:** Initiate the model training process.
4.  **Download and Test:** Upon completion, you can download the trained model and scaler files, and immediately test the new model with a separate dataset.

Detailed steps for the model training interface are available in :doc:`app`.