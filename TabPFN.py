import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler  # Import StandardScaler
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tabpfn import TabPFNRegressor
import seaborn as sns
import shap

plt.rcParams['font.family'] = 'Times New Roman'

# Load data
df = pd.read_csv('')
X = df[['']].values  # Convert to ndarray for easier indexing
y = df[''].values

# Initialize KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Store results for each fold
metrics = {
    'train': {'rmse': [], 'r2': [], 'mae': [], 'rpd': [], 'corr': []},
    'val': {'rmse': [], 'r2': [], 'mae': [], 'rpd': [], 'corr': []}
}

# Collect all predictions for final plotting
y_train_all, y_pred_train_all = [], []
y_val_all, y_pred_val_all = [], []

# 5-Fold CV loop
for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Initialize TabPFN model (trained independently for each fold)
    model = TabPFNRegressor(device='cpu')  # Can add device='cuda' if GPU is available
    model.fit(X_train, y_train)

    # Predict
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)


    # Calculate metrics
    def calc_metrics(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        std = np.std(y_true)
        rpd = std / rmse if rmse != 0 else np.nan
        corr, _ = pearsonr(y_true, y_pred)
        return rmse, r2, mae, rpd, corr


    tr_rmse, tr_r2, tr_mae, tr_rpd, tr_corr = calc_metrics(y_train, y_pred_train)
    val_rmse, val_r2, val_mae, val_rpd, val_corr = calc_metrics(y_val, y_pred_val)

    # Store
    metrics['train']['rmse'].append(tr_rmse)
    metrics['train']['r2'].append(tr_r2)
    metrics['train']['mae'].append(tr_mae)
    metrics['train']['rpd'].append(tr_rpd)
    metrics['train']['corr'].append(tr_corr)

    metrics['val']['rmse'].append(val_rmse)
    metrics['val']['r2'].append(val_r2)
    metrics['val']['mae'].append(val_mae)
    metrics['val']['rpd'].append(val_rpd)
    metrics['val']['corr'].append(val_corr)

    # Collect aggregated predictions (for plotting)
    y_train_all.extend(y_train)
    y_pred_train_all.extend(y_pred_train)
    y_val_all.extend(y_val)
    y_pred_val_all.extend(y_pred_val)

    print(f"Fold {fold} - Val R²: {val_r2:.4f}, RMSE: {val_rmse:.4f}")


# Calculate mean ± standard deviation
def mean_std(lst):
    return np.mean(lst), np.std(lst)


print("\n=== 5-Fold Cross-Validation Results (Mean ± Std) ===")
for name in ['train', 'val']:
    print(f"\n{name.capitalize()} Set:")
    for metric in ['rmse', 'r2', 'mae', 'rpd', 'corr']:
        m, s = mean_std(metrics[name][metric])
        print(f"  {metric.upper()}: {m:.4f} ± {s:.4f}")

# ------------------------------
# Visualization: Aggregate predictions vs actual across all folds (more robust)
plt.figure(figsize=(12, 5))

# Training set (all folds combined)
plt.subplot(1, 2, 1)
plt.scatter(y_train_all, y_pred_train_all, alpha=0.5, s=20, label='Predictions')
plt.plot([min(y_train_all), max(y_train_all)], [min(y_train_all), max(y_train_all)], 'r--', label='Perfect Fit')
plt.xlabel('Actual EC')
plt.ylabel('Predicted EC')
mean_r2_train = np.mean(metrics['train']['r2'])
plt.title(f'Training (All Folds, Avg R² = {mean_r2_train:.4f})')
plt.legend()

# Validation set (all folds combined)
plt.subplot(1, 2, 2)
plt.scatter(y_val_all, y_pred_val_all, alpha=0.6, s=25, color='C1')
plt.plot([min(y_val_all), max(y_val_all)], [min(y_val_all), max(y_val_all)], 'r--')
plt.xlabel('Actual EC')
plt.ylabel('Predicted EC')
mean_r2_val = np.mean(metrics['val']['r2'])
plt.title(f'Validation (All Folds, Avg R² = {mean_r2_val:.4f})')

plt.tight_layout()
plt.show()