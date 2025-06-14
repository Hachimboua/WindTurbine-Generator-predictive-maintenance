Models Testing Documentation
============================

Overview
--------
This notebook tests four different neural network models for a single output prediction task:
1. BiLSTM
2. BiGRU 
3. Conv1D_LSTM
4. LSTM_Attention

Model Training Process
----------------------
Each model was trained with:
- Early stopping to prevent overfitting
- 500 maximum epochs  
- Training and validation loss tracking
- Best model saved (`.pth` files)
- Scaler saved (`.joblib` files)

Performance Comparison
----------------------
| Model           | R² Score  | MSE      |
|-----------------|----------|----------|
| BiLSTM          | 0.991752 | 0.000339 |
| BiGRU           | 0.991519 | 0.000349 |
| Conv1D_LSTM     | 0.991491 | 0.000350 | 
| LSTM_Attention  | 0.979971 | 0.000824 |

Key Observations
----------------
- All models achieved high R² scores (>0.97)
- BiLSTM performed slightly better than others  
- Training stopped early (12-48 epochs)
- Includes final comparison plots

Saved Files
-----------
For each model:
- Model weights (`.pth`)
- Scaler object (`.joblib`)

Usage Notes
-----------
- Trained on CUDA-enabled hardware
- Includes training progress visualizations
- Early stopping based on validation loss