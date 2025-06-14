import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import plotly.graph_objects as go
import os
import io
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# 1. APP CONFIGURATION
st.set_page_config(page_title="Direct Forecast Dashboard", page_icon="⚡", layout="wide")

# 2. CONFIG & MODEL DEFINITIONS

# --- General Model Configuration ---
MODEL_CONFIG = {
    "window_size": 50,
    "future_steps": 1200,
    "target_columns": ["degradation", "temperature_estimated", "time_since_last_maintenance"],
    "degradation_threshold": 0.66,
    # RNN-specific
    "hidden_size": 64,
    "num_layers": 3,
    "dropout": 0.3,
    # Conv1D-specific
    "conv_filters": 64,
    "kernel_size": 3,
    # Training-specific
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "batch_size": 32,
    "num_epochs": 50, # A default value for the UI
    "validation_split": 0.2,
}
MODEL_CONFIG["output_size"] = len(MODEL_CONFIG["target_columns"]) * MODEL_CONFIG["future_steps"]

# --- Model Architectures ---

class BiLSTMModel(nn.Module):
    """A Bidirectional LSTM model for direct forecasting."""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout, **kwargs):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

class BiGRUModel(nn.Module):
    """A Bidirectional GRU model for direct forecasting."""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout, **kwargs):
        super(BiGRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        return self.fc(gru_out[:, -1, :])

class Conv1DLSTMModel(nn.Module):
    """A 1D Conv layer followed by a unidirectional LSTM."""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout, conv_filters, kernel_size, **kwargs):
        super(Conv1DLSTMModel, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=conv_filters, kernel_size=kernel_size, padding='same')
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(conv_filters, hidden_size, num_layers, batch_first=True, bidirectional=False, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1d(x))
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# --- Mapping user choice to model files and classes ---
MODELS_AVAILABLE = {
    "BiLSTM": {"class": BiLSTMModel, "path": "Dashboard_App/model_BiLSTM.pth"},
    "BiGRU": {"class": BiGRUModel, "path": "Dashboard_App/model_BiGRU.pth"},
    "Conv1D-LSTM": {"class": Conv1DLSTMModel, "path": "Dashboard_App/model_Conv1D_LSTM.pth"}
}

# 3. DATA PREPARATION HELPERS (FOR TRAINING)

def create_direct_sequences(data, target_column_indices, window_size, future_steps):
    """Creates sequences where X is the history and y is the entire future trajectory, flattened."""
    X, y = [], []
    num_samples = len(data) - window_size - future_steps + 1
    if num_samples <= 0:
        raise ValueError("Not enough data to create sequences with the given window_size and future_steps.")
    for i in range(num_samples):
        X.append(data[i : i + window_size, :])
        future_sequence = data[i + window_size : i + window_size + future_steps, target_column_indices]
        y.append(future_sequence.flatten())
    return np.array(X), np.array(y)

class TimeSeriesDataset(Dataset):
    """A custom PyTorch Dataset for time series sequence data."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 4. CORE TRAINING FUNCTION

def run_training(model_class, df_train, config):
    """
    Orchestrates the model training process from start to finish.
    Returns the best model state dictionary and the fitted scaler.
    """
    try:
        # Step 1: Preprocessing and Scaling
        st.write("### Step 1: Preprocessing data and fitting scaler...")
        scaler = StandardScaler()
        df_numeric = df_train.select_dtypes(include=np.number)
        scaled_data = scaler.fit_transform(df_numeric)
        st.success(f"Scaler fitted successfully on {len(scaler.feature_names_in_)} features.")

        # Step 2: Create Sequences
        st.write("### Step 2: Creating training sequences...")
        target_indices = [df_numeric.columns.get_loc(c) for c in config["target_columns"]]
        X, y = create_direct_sequences(scaled_data, target_indices, config["window_size"], config["future_steps"])
        st.write(f"Created {len(X)} samples.")

        # Step 3: Split data and create DataLoaders
        st.write("### Step 3: Splitting data and creating DataLoaders...")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=config["validation_split"], random_state=42)
        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

        # Step 4: Initialize Model, Optimizer, Loss
        st.write("### Step 4: Initializing model, optimizer, and loss function...")
        model_constructor_config = {"input_size": X_train.shape[2], **config}
        model = model_class(**model_constructor_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        criterion = nn.MSELoss()

        # Step 5: Training Loop
        st.write("### Step 5: Starting model training...")
        status_text = st.empty()
        progress_bar = st.progress(0)
        loss_plot_placeholder = st.empty()
        train_losses, val_losses = [], []
        best_val_loss = float('inf')
        best_model_state = None

        for epoch in range(config["num_epochs"]):
            model.train()
            epoch_train_loss = sum(criterion(model(batch_X), batch_y).item() for batch_X, batch_y in train_loader)
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            model.eval()
            with torch.no_grad():
                epoch_val_loss = sum(criterion(model(batch_X), batch_y).item() for batch_X, batch_y in val_loader)
            avg_val_loss = epoch_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            status_text.text(f"Epoch {epoch+1}/{config['num_epochs']} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
            progress_bar.progress((epoch + 1) / config["num_epochs"])

            fig = go.Figure()
            fig.add_trace(go.Scatter(y=train_losses, mode='lines+markers', name='Training Loss'))
            fig.add_trace(go.Scatter(y=val_losses, mode='lines+markers', name='Validation Loss'))
            fig.update_layout(title="Training & Validation Loss Over Epochs", xaxis_title="Epoch", yaxis_title="MSE Loss")
            loss_plot_placeholder.plotly_chart(fig, use_container_width=True)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict()

        st.success(f"**Training complete!** Best validation loss: {best_val_loss:.6f}")
        return best_model_state, scaler

    except Exception as e:
        st.error(f"An error occurred during training: {e}")
        return None, None


# 5. CACHED HELPER FUNCTIONS (FOR FORECASTING)

@st.cache_resource
def load_model_and_scaler(model_path, scaler_path, model_class, config):
    """Loads a model and scaler for forecasting."""
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.sidebar.error(f"Model ({model_path}) or scaler not found.")
        return None, None
    scaler = joblib.load(scaler_path)
    model_constructor_config = {"input_size": len(scaler.feature_names_in_), **config}
    model = model_class(**model_constructor_config)
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    except RuntimeError as e:
        st.error(f"Error loading model state. Architecture may not match. Details: {e}")
        return None, None
    model.eval()
    return model, scaler

@st.cache_data
def load_data(uploaded_file, timestamp_col="time"): # Default to "time"
    """Loads data from an uploaded CSV file, optionally parsing a timestamp column."""
    try:
        parse_dates = [timestamp_col] if timestamp_col else False
        df = pd.read_csv(uploaded_file, parse_dates=parse_dates)
        if timestamp_col and timestamp_col in df.columns:
            df.set_index(timestamp_col, inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading or parsing data: {e}")
        return None


# 6. CORE FORECASTING FUNCTION

def run_direct_forecast(model, scaler, history_window_df, config):
    """Runs a forecast using a loaded model and scaler."""
    history_numeric = history_window_df.select_dtypes(include=[np.number])
    history_aligned = history_numeric.reindex(columns=scaler.feature_names_in_, fill_value=0)
    history_scaled = scaler.transform(history_aligned)
    input_tensor = torch.tensor(history_scaled, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        flat_prediction_scaled = model(input_tensor)

    num_targets = len(config["target_columns"])
    prediction_scaled = flat_prediction_scaled.cpu().numpy().reshape(config["future_steps"], num_targets)
    prediction_full_features = np.zeros((config["future_steps"], len(scaler.feature_names_in_)))
    target_indices = [list(scaler.feature_names_in_).index(c) for c in config["target_columns"]]
    prediction_full_features[:, target_indices] = prediction_scaled
    prediction_unscaled = scaler.inverse_transform(prediction_full_features)
    
    return pd.DataFrame(prediction_unscaled[:, target_indices], columns=config["target_columns"])

# 7. STREAMLIT UI

st.sidebar.title("Dashboard Controls")
page = st.sidebar.selectbox("Choose a Page", ["Welcome", "Live RUL Prediction", "Component Health Forecasting", "Model Training"])

# --- Welcome Page ---
if page == "Welcome":
    st.title("⚡ Welcome to the Direct Forecast Dashboard!")
    st.markdown("This application allows you to **forecast component health**, determine **Remaining Useful Life (RUL)**, and **re-train predictive models** on your own data.")
    st.markdown("---")

    with st.container(border=True):
        st.header("🏁 Get Started in 3 Steps")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### 1. Select a Page")
            st.markdown("Use the dropdown in the sidebar to navigate between forecasting and training pages.")
        with col2:
            st.markdown("#### 2. Choose a Model")
            st.markdown("Select a model architecture (`BiLSTM`, `BiGRU`, etc.) from the sidebar.")
        with col3:
            st.markdown("#### 3. Upload Your Data")
            st.markdown("Provide your historical component data as a CSV file to run a forecast or start training.")
    
    st.markdown("---")
    
    st.header("⚙️ How It Works")
    colA, colB = st.columns(2)
    with colA:
        with st.container(border=True):
            st.subheader("🔮 Forecasting")
            st.markdown("- **Live RUL Prediction:** Forecasts a component's degradation to predict its end-of-life in operating hours.")
            st.markdown("- **Component Health Forecasting:** Provides future trajectories for multiple health indicators at once.")
    with colB:
        with st.container(border=True):
            st.subheader("🏋️ Model Training")
            st.markdown("- Upload your own time-series data.")
            st.markdown("- Configure hyperparameters like epochs and learning rate.")
            st.markdown("- Train a new model and download the resulting model (`.pth`) and scaler (`.joblib`) files to use locally.")
    
    st.info("To begin, select a page from the sidebar menu.", icon="👈")


# --- Model Training Page ---
elif page == "Model Training":
    st.title("🏋️ Model Training")
    st.markdown("Re-train a model on your own data. After training, you can download the new model and data scaler files to your local machine.")
    
    st.sidebar.header("Training Parameters")
    model_choice_train = st.sidebar.selectbox("Choose Model Architecture", options=list(MODELS_AVAILABLE.keys()), key="train_model_choice")
    
    MODEL_CONFIG["num_epochs"] = st.sidebar.number_input("Number of Epochs", min_value=1, max_value=1000, value=50)
    MODEL_CONFIG["learning_rate"] = st.sidebar.number_input("Learning Rate", min_value=1e-6, value=0.001, format="%.5f")
    MODEL_CONFIG["batch_size"] = st.sidebar.number_input("Batch Size", min_value=1, value=32)
    
    uploaded_train_file = st.file_uploader("Upload Your Full Training Data (CSV)", type="csv")
    
    if "trained_model_state" not in st.session_state:
        st.session_state.trained_model_state = None
    if "trained_scaler" not in st.session_state:
        st.session_state.trained_scaler = None

    if uploaded_train_file:
        # Timestamp column hardcoded to "time" for training data
        df_for_training = load_data(uploaded_train_file, timestamp_col="time")
        if df_for_training is not None:
            st.write("Data Preview:")
            st.dataframe(df_for_training.head())
            
            if st.button(f"Start Training for {model_choice_train}", type="primary"):
                st.session_state.trained_model_state, st.session_state.trained_scaler = run_training(
                    model_class=MODELS_AVAILABLE[model_choice_train]['class'],
                    df_train=df_for_training,
                    config=MODEL_CONFIG,
                )
    
    if st.session_state.trained_model_state and st.session_state.trained_scaler:
        with st.container(border=True):
            st.header("Post-Training Actions")
            
            st.subheader("1. Download New Files Locally")
            # Serialize model to in-memory file
            model_buffer = io.BytesIO()
            # Need to initialize a model instance to save its state dict correctly
            model_class_instance = MODELS_AVAILABLE[model_choice_train]['class'](input_size=len(st.session_state.trained_scaler.feature_names_in_), **MODEL_CONFIG)
            model_class_instance.load_state_dict(st.session_state.trained_model_state)
            torch.save(model_class_instance.state_dict(), model_buffer)
            model_buffer.seek(0)

            # Serialize scaler to in-memory file
            scaler_buffer = io.BytesIO()
            joblib.dump(st.session_state.trained_scaler, scaler_buffer)
            scaler_buffer.seek(0)
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📥 Download Model (.pth)",
                    data=model_buffer,
                    file_name=f"trained_{model_choice_train}.pth",
                    mime="application/octet-stream"
                )
            with col2:
                st.download_button(
                    label="📥 Download Scaler (.joblib)",
                    data=scaler_buffer,
                    file_name="trained_scaler.joblib",
                    mime="application/octet-stream"
                )
            
            st.subheader("2. Test New Model with a Quick Forecast (No Overwrite)")
            st.info("Upload a history CSV to see how your newly trained model performs without replacing the original files.")
            uploaded_test_file = st.file_uploader("Upload History Data for Testing (CSV)", type="csv", key="test_new_model_file_uploader")
            if uploaded_test_file:
                # Timestamp column hardcoded to "time" for test data
                df_test_history = load_data(uploaded_test_file, timestamp_col="time")
                if df_test_history is not None and len(df_test_history) > MODEL_CONFIG["window_size"]:
                    test_prediction_point = st.slider("Select Forecast Starting Point for Test (by index)", MODEL_CONFIG["window_size"], len(df_test_history) - 1, len(df_test_history) - 1, key="test_pred_point")
                    targets_to_plot_test = st.multiselect("Select Targets to Plot for Test", options=MODEL_CONFIG["target_columns"], default=MODEL_CONFIG["target_columns"], key="test_targets_to_plot")
                    
                    if st.button("Run Test Forecast with New Model", key="run_test_forecast_button"):
                        with st.spinner("Running test forecast with the newly trained model..."):
                            # Create a temporary model instance and load the trained state dict
                            temp_model = MODELS_AVAILABLE[model_choice_train]['class'](input_size=len(st.session_state.trained_scaler.feature_names_in_), **MODEL_CONFIG)
                            temp_model.load_state_dict(st.session_state.trained_model_state)
                            temp_model.eval()

                            history_window_df_test = df_test_history.iloc[test_prediction_point - MODEL_CONFIG["window_size"]:test_prediction_point]
                            forecast_df_test = run_direct_forecast(temp_model, st.session_state.trained_scaler, history_window_df_test, MODEL_CONFIG)
                            st.success("Test Forecast Complete!")

                            if isinstance(df_test_history.index, pd.DatetimeIndex):
                                forecast_start_time_test = df_test_history.index[test_prediction_point]
                                forecast_index_test = pd.date_range(start=forecast_start_time_test, periods=len(forecast_df_test), freq='H')
                            else:
                                forecast_index_test = np.arange(test_prediction_point, test_prediction_point + len(forecast_df_test))
                            
                            forecast_df_test.index = forecast_index_test
                            st.dataframe(forecast_df_test.head())

                            for target in targets_to_plot_test:
                                fig_test = go.Figure()
                                fig_test.add_trace(go.Scatter(x=df_test_history.index, y=df_test_history[target], name=f'Historical {target}', line=dict(color='royalblue')))
                                fig_test.add_trace(go.Scatter(x=forecast_df_test.index, y=forecast_df_test[target], name=f'Forecasted {target}', line=dict(color='red', dash='dash')))
                                
                                vline_x_test = df_test_history.index[test_prediction_point] if isinstance(df_test_history.index, pd.DatetimeIndex) else test_prediction_point
                                
                                if isinstance(df_test_history.index, pd.DatetimeIndex):
                                    fig_test.add_vline(x=vline_x_test, line_dash="dot", line_color="green")
                                    fig_test.add_annotation(x=vline_x_test, y=1, yref="paper", showarrow=False, text="Forecast Start", font=dict(color="green"), xanchor="right", yanchor="bottom", yshift=5)
                                else:
                                    fig_test.add_vline(x=vline_x_test, line_dash="dot", line_color="green", annotation_text="Forecast Start")
                                
                                # Add RUL/Failure point specifically for 'degradation' target in the test plot
                                if target == "degradation":
                                    forecasted_degradation_test = forecast_df_test['degradation'].values
                                    predicted_rul_test = -1
                                    try:
                                        predicted_rul_test = np.where(forecasted_degradation_test >= MODEL_CONFIG["degradation_threshold"])[0][0]
                                    except IndexError:
                                        pass # Degradation threshold not reached within forecast window

                                    if predicted_rul_test != -1:
                                        failure_point_x_test = forecast_index_test[predicted_rul_test] if isinstance(forecast_index_test, pd.DatetimeIndex) else test_prediction_point + predicted_rul_test
                                        fig_test.add_trace(go.Scatter(x=[failure_point_x_test], y=[forecast_df_test['degradation'].iloc[predicted_rul_test]], mode='markers', marker=dict(color='purple', size=15, symbol='circle'), name=f'Predicted Failure (RUL: {predicted_rul_test} hrs)'))
                                    fig_test.add_hline(y=MODEL_CONFIG["degradation_threshold"], line_dash="dot", line_color="orange", annotation_text="Failure Threshold")
                                    
                                fig_test.update_layout(title=f"<b>TEST: {target.replace('_', ' ').title()} - History and Forecast with New Model</b>", xaxis_title="Timestamp", yaxis_title="Value")
                                st.plotly_chart(fig_test, use_container_width=True)

            st.subheader("3. Overwrite Original Files on Server")
            st.warning("This action cannot be undone. It will replace the current model and scaler used by the forecasting pages.", icon="⚠️")
            if st.button(f"Update original **{model_choice_train}** files", type="primary"):
                try:
                    # Ensure directory exists
                    os.makedirs("Dashboard_App", exist_ok=True)
                    model_path = MODELS_AVAILABLE[model_choice_train]['path']
                    scaler_path = "Dashboard_App/main_scaler.joblib"
                    
                    # Save the model state and scaler
                    # Re-initialize the model to save the state dict properly
                    model_to_save = MODELS_AVAILABLE[model_choice_train]['class'](input_size=len(st.session_state.trained_scaler.feature_names_in_), **MODEL_CONFIG)
                    model_to_save.load_state_dict(st.session_state.trained_model_state)
                    torch.save(model_to_save.state_dict(), model_path)
                    joblib.dump(st.session_state.trained_scaler, scaler_path)
                    
                    st.success(f"Successfully updated and saved:\n- `{model_path}`\n- `{scaler_path}`")
                    st.info("The forecasting pages will now use these new files. You may need to clear the cache and reload the page for changes to take full effect.")
                    # Clear the cached load function to force a reload on the next page visit
                    st.cache_resource.clear()

                except Exception as e:
                    st.error(f"Failed to save files: {e}")


# --- Forecasting Pages ---
else:
    st.title(f"⚡ {page}")
    st.sidebar.header("Forecast Settings")
    model_choice = st.sidebar.selectbox("Choose a Model", options=list(MODELS_AVAILABLE.keys()))
    
    # Removed timestamp_col_name input
    # timestamp_col_name = st.sidebar.text_input("Timestamp Column Name", value="time", help="Enter the exact name of the timestamp column in your CSV.")
    
    selected_model_info = MODELS_AVAILABLE[model_choice]
    model, scaler = load_model_and_scaler(
        model_path=selected_model_info["path"],
        scaler_path="Dashboard_App/main_scaler.joblib",
        model_class=selected_model_info["class"],
        config=MODEL_CONFIG
    )

    if model and scaler:
        uploaded_file = st.sidebar.file_uploader("Upload Full History CSV", type="csv")
        if uploaded_file:
            # Pass the user-provided column name to the loading function, hardcoded to "time"
            df_history = load_data(uploaded_file, timestamp_col="time")
            
            if df_history is not None and len(df_history) > MODEL_CONFIG["window_size"]:
                max_point = len(df_history) - 1
                min_point = MODEL_CONFIG["window_size"]
                prediction_point = st.sidebar.slider("Select Forecast Starting Point (by index)", min_point, max_point, max_point, help=f"The model uses the {MODEL_CONFIG['window_size']} steps before this point.")

                if page == "Live RUL Prediction":
                    st.markdown("Derive the **Remaining Useful Life (RUL)** by forecasting the degradation curve until it hits the failure threshold.")
                    if st.sidebar.button("Calculate RUL", key="rul_button", type="primary"):
                        with st.spinner(f"Forecasting with {model_choice}..."):
                            history_window_df = df_history.iloc[prediction_point - MODEL_CONFIG["window_size"]:prediction_point]
                            forecast_df = run_direct_forecast(model, scaler, history_window_df, MODEL_CONFIG)
                            forecasted_degradation = forecast_df['degradation'].values
                            try:
                                predicted_rul = np.where(forecasted_degradation >= MODEL_CONFIG["degradation_threshold"])[0][0]
                            except IndexError:
                                predicted_rul = -1
                            st.success("Analysis Complete!")
                            if predicted_rul != -1:
                                st.metric(label=f"Predicted RUL from selected point", value=f"{predicted_rul} Hours")
                            else:
                                st.warning(f"The component is not predicted to reach the failure threshold within the next {MODEL_CONFIG['future_steps']} hours. Its RUL is greater than the forecast window.", icon="ℹ️")

                            fig = go.Figure()
                            # Use datetime index for plotting if available
                            if isinstance(df_history.index, pd.DatetimeIndex):
                                forecast_start_time = df_history.index[prediction_point]
                                forecast_index = pd.date_range(start=forecast_start_time, periods=len(forecast_df), freq='H')
                                failure_point_x = forecast_start_time + pd.to_timedelta(predicted_rul, unit='h') if predicted_rul != -1 else None
                                vline_x = forecast_start_time
                            else: # Fallback to integer index
                                forecast_index = np.arange(prediction_point, prediction_point + len(forecast_df))
                                failure_point_x = prediction_point + predicted_rul if predicted_rul != -1 else None
                                vline_x = prediction_point

                            fig.add_trace(go.Scatter(x=df_history.index, y=df_history['degradation'], name='Historical Degradation', line=dict(color='royalblue')))
                            fig.add_trace(go.Scatter(x=forecast_index, y=forecast_df['degradation'], name='Forecasted Degradation', line=dict(color='red', dash='dash')))
                            if predicted_rul != -1 and failure_point_x:
                                fig.add_trace(go.Scatter(x=[failure_point_x], y=[forecast_df['degradation'].iloc[predicted_rul]], mode='markers', marker=dict(color='purple', size=15, symbol='circle'), name=f'Predicted Failure (RUL: {predicted_rul} hrs)')) # Changed symbol for consistency
                            fig.add_hline(y=MODEL_CONFIG["degradation_threshold"], line_dash="dot", line_color="orange", annotation_text="Failure Threshold")
                            
                            # Conditional vline and annotation to avoid TypeError
                            if isinstance(df_history.index, pd.DatetimeIndex):
                                fig.add_vline(x=vline_x, line_dash="dot", line_color="green")
                                fig.add_annotation(x=vline_x, y=1, yref="paper", showarrow=False, text="Forecast Start", font=dict(color="green"), xanchor="right", yanchor="bottom", yshift=5)
                            else:
                                fig.add_vline(x=vline_x, line_dash="dot", line_color="green", annotation_text="Forecast Start")

                            fig.update_layout(title="<b>Degradation History and Forecast</b>", xaxis_title="Timestamp", yaxis_title="Degradation")
                            st.plotly_chart(fig, use_container_width=True)

                elif page == "Component Health Forecasting":
                    st.markdown("Forecast the future trajectory of all target components in a single prediction.")
                    targets_to_plot = st.sidebar.multiselect("Select Targets to Plot", options=MODEL_CONFIG["target_columns"], default=MODEL_CONFIG["target_columns"])
                    if st.sidebar.button("Forecast Components", key="health_button", type="primary"):
                        with st.spinner(f"Forecasting all components with {model_choice}..."):
                            history_window_df = df_history.iloc[prediction_point - MODEL_CONFIG["window_size"]:prediction_point]
                            forecast_df = run_direct_forecast(model, scaler, history_window_df, MODEL_CONFIG)
                            st.success("Forecast Complete!")
                            
                            # Create forecast index for plotting
                            if isinstance(df_history.index, pd.DatetimeIndex):
                                forecast_start_time = df_history.index[prediction_point]
                                forecast_index = pd.date_range(start=forecast_start_time, periods=len(forecast_df), freq='H')
                            else: # Fallback to integer index
                                forecast_index = np.arange(prediction_point, prediction_point + len(forecast_df))

                            # Assign the new index to the forecast dataframe for easier plotting
                            forecast_df.index = forecast_index
                            st.dataframe(forecast_df.head())

                            for target in targets_to_plot:
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(x=df_history.index, y=df_history[target], name=f'Historical {target}', line=dict(color='royalblue')))
                                fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df[target], name=f'Forecasted {target}', line=dict(color='red', dash='dash')))
                                
                                vline_x = df_history.index[prediction_point] if isinstance(df_history.index, pd.DatetimeIndex) else prediction_point
                                
                                # Conditional vline and annotation to avoid TypeError
                                if isinstance(df_history.index, pd.DatetimeIndex):
                                    fig.add_vline(x=vline_x, line_dash="dot", line_color="green")
                                    fig.add_annotation(x=vline_x, y=1, yref="paper", showarrow=False, text="Forecast Start", font=dict(color="green"), xanchor="right", yanchor="bottom", yshift=5)
                                else:
                                    fig.add_vline(x=vline_x, line_dash="dot", line_color="green", annotation_text="Forecast Start")
                                
                                fig.update_layout(title=f"<b>{target.replace('_', ' ').title()}: History and Direct Forecast</b>", xaxis_title="Timestamp", yaxis_title="Value")
                                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Please upload a data file with enough history (at least `window_size` + 1 data points) and select a model using the sidebar to begin.")
        else:
            st.info("Please upload a data file and select a model using the sidebar to begin.")
    else:
        st.error("Could not load the selected model and/or scaler. Please check the file paths and ensure the correct files exist in the `Dashboard_App` directory.")