Model Architectures and Training
================================

This section provides an overview of the deep learning models used in the Predictive Maintenance AI Dashboard and details the process of training new models.

Model Architectures
-------------------

The dashboard supports the following deep learning architectures, suitable for time-series forecasting:

* **BiLSTM (Bidirectional Long Short-Term Memory):**
    * A type of Recurrent Neural Network (RNN) designed to learn long-term dependencies in sequence data.
    * "Bidirectional" means it processes the sequence both forwards and backward, capturing context from both past and future elements.
    * Key parameters: `input_size`, `hidden_size`, `num_layers`, `output_size`, `dropout`.

* **BiGRU (Bidirectional Gated Recurrent Unit):**
    * Similar to BiLSTM but with a simpler gating mechanism, making it computationally less intensive while often achieving comparable performance.
    * Also processes sequences bidirectionally.
    * Key parameters: `input_size`, `hidden_size`, `num_layers`, `output_size`, `dropout`.

* **Conv1D-LSTM (1D Convolutional LSTM):**
    * Combines a 1D Convolutional layer with an LSTM layer.
    * The **1D Convolutional layer** (`nn.Conv1d`) is effective at extracting local features and patterns from the input sequence (e.g., short-term trends or anomalies).
    * The output of the convolutional layer is then fed into an **LSTM layer** to capture sequential dependencies over time.
    * Offers a hybrid approach, useful for data with both local and long-range temporal patterns.
    * Key parameters: `input_size`, `hidden_size`, `num_layers`, `output_size`, `dropout`, `conv_filters`, `kernel_size`.

Training Process
----------------

The "Train New Model" page allows you to train any of the above architectures on your custom dataset.

1.  **Data Preparation:**
    * Your uploaded CSV data is first processed by :doc:`preprocessing` (scaling) and then converted into input-output sequences using the `create_direct_sequences` function. This function creates pairs of historical windows (X) and corresponding future target sequences (y).
    * The data is then split into 90% training and 10% validation sets.
2.  **Model Initialization:**
    * The selected model architecture is initialized with configurable hyperparameters (hidden size, number of layers, dropout).
    * The `output_size` is determined by `len(target_columns) * future_steps`, reflecting the flattened prediction for all target variables over the `future_steps` horizon.
3.  **Optimization:**
    * **Optimizer:** `torch.optim.Adam` is used for optimization.
    * **Loss Function:** `nn.MSELoss()` (Mean Squared Error) is used to measure the difference between the model's predictions and the actual future values.
4.  **Training Loop:**
    * The model iterates through the training data for a specified number of `epochs`.
    * In each epoch, data is loaded in batches, predictions are made, the loss is calculated, and model weights are updated via backpropagation.
    * Validation loss is calculated after each epoch to monitor overfitting.
5.  **Live Progress Visualization:**
    * During training, the dashboard displays live updates of training and validation loss, allowing you to monitor the model's learning progress.

After training, you can download the `.pth` file (containing the model's learned weights) and the `.joblib` file (containing the fitted `StandardScaler`). These files are essential for using your newly trained model for future forecasts without re-training.