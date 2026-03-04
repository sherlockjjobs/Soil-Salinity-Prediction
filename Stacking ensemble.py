import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor
import scipy.stats as stats

warnings.filterwarnings("ignore")

# Set global font to Times New Roman (scientific publication requirement)
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
sns.set(style="whitegrid", font="Times New Roman")

# Load data
df = pd.read_excel('')  # Please ensure the filename matches your actual data file

# Split features and target variable
X = df.drop([''], axis=1)
y = df['']

# Split training and testing sets (fix random_state for reproducibility)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42  # ← Fix random_state=42
)

# ========================
# 1. KNN Model Tuning and Evaluation
# ========================
print("Fitting KNN GridSearchCV...")
knn_model = KNeighborsRegressor()
knn_param_grid = {
    'n_neighbors': list(range(1, 31)),
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}
knn_cv = KFold(n_splits=5, shuffle=True, random_state=42)
knn_grid_search = GridSearchCV(
    estimator=knn_model,
    param_grid=knn_param_grid,
    cv=knn_cv,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)
knn_grid_search.fit(X_train, y_train)
print("KNN Best parameters found: ", knn_grid_search.best_params_)
knn_best_model = knn_grid_search.best_estimator_
knn_y_pred = knn_best_model.predict(X_test)

# Test set metrics (unique variable names)
knn_rmse_test = np.sqrt(mean_squared_error(y_test, knn_y_pred))
knn_mae_test = mean_absolute_error(y_test, knn_y_pred)
knn_r2_test = r2_score(y_test, knn_y_pred)
knn_rpd_test = np.std(y_test) / knn_rmse_test if knn_rmse_test != 0 else np.inf
knn_pearson_test, _ = pearsonr(y_test, knn_y_pred)

print(f"KNN Test RMSE: {knn_rmse_test:.4f}, R²: {knn_r2_test:.4f}, MAE: {knn_mae_test:.4f}")

# ========================
# 2. Random Forest
# ========================
print("Fitting Random Forest GridSearchCV...")
rf_model = RandomForestRegressor(random_state=42)
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}
rf_cv = KFold(n_splits=5, shuffle=True, random_state=42)
rf_grid_search = GridSearchCV(rf_model, rf_param_grid, cv=rf_cv,
                              scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
rf_grid_search.fit(X_train, y_train)
print("RF Best parameters found: ", rf_grid_search.best_params_)
rf_best_model = rf_grid_search.best_estimator_
rf_y_pred = rf_best_model.predict(X_test)

rf_rmse_test = np.sqrt(mean_squared_error(y_test, rf_y_pred))
rf_mae_test = mean_absolute_error(y_test, rf_y_pred)
rf_r2_test = r2_score(y_test, rf_y_pred)
rf_rpd_test = np.std(y_test) / rf_rmse_test if rf_rmse_test != 0 else np.inf
rf_pearson_test, _ = pearsonr(y_test, rf_y_pred)

print(f"RF Test RMSE: {rf_rmse_test:.4f}, R²: {rf_r2_test:.4f}, MAE: {rf_mae_test:.4f}")

# ========================
# 3. XGBoost
# ========================
print("Fitting XGBoost GridSearchCV...")
xgb_model = XGBRegressor(random_state=42, verbosity=0)
xgb_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1],
    'min_child_weight': [1, 5]
}
xgb_cv = KFold(n_splits=5, shuffle=True, random_state=42)
xgb_grid_search = GridSearchCV(xgb_model, xgb_param_grid, cv=xgb_cv,
                               scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
xgb_grid_search.fit(X_train, y_train)
print("XGBoost Best parameters found: ", xgb_grid_search.best_params_)
xgb_best_model = xgb_grid_search.best_estimator_
xgb_y_pred = xgb_best_model.predict(X_test)

xgb_rmse_test = np.sqrt(mean_squared_error(y_test, xgb_y_pred))
xgb_mae_test = mean_absolute_error(y_test, xgb_y_pred)
xgb_r2_test = r2_score(y_test, xgb_y_pred)
xgb_rpd_test = np.std(y_test) / xgb_rmse_test if xgb_rmse_test != 0 else np.inf
xgb_pearson_test, _ = pearsonr(y_test, xgb_y_pred)

print(f"XGBoost Test RMSE: {xgb_rmse_test:.4f}, R²: {xgb_r2_test:.4f}, MAE: {xgb_mae_test:.4f}")

# ========================
# 4. LightGBM
# ========================
print("Fitting LGBM GridSearchCV...")
lgbm_model = LGBMRegressor(random_state=42, verbose=-1)
lgbm_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 10],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'min_child_samples': [10, 20],
    'reg_alpha': [0, 0.1],
    'reg_lambda': [0, 0.1]
}
lgbm_cv = KFold(n_splits=5, shuffle=True, random_state=42)
lgbm_grid_search = GridSearchCV(lgbm_model, lgbm_param_grid, cv=lgbm_cv,
                                scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
lgbm_grid_search.fit(X_train, y_train)
print("LGBM Best parameters found: ", lgbm_grid_search.best_params_)
lgbm_best_model = lgbm_grid_search.best_estimator_
lgbm_y_pred = lgbm_best_model.predict(X_test)

lgbm_rmse_test = np.sqrt(mean_squared_error(y_test, lgbm_y_pred))
lgbm_mae_test = mean_absolute_error(y_test, lgbm_y_pred)
lgbm_r2_test = r2_score(y_test, lgbm_y_pred)
lgbm_rpd_test = np.std(y_test) / lgbm_rmse_test if lgbm_rmse_test != 0 else np.inf
lgbm_pearson_test, _ = pearsonr(y_test, lgbm_y_pred)

print(f"LGBM Test RMSE: {lgbm_rmse_test:.4f}, R²: {lgbm_r2_test:.4f}, MAE: {lgbm_mae_test:.4f}")

# ========================
# 5. CatBoost
# ========================
print("Fitting CatBoost GridSearchCV...")
catboost_model = CatBoostRegressor(random_state=42, verbose=0)
catboost_param_grid = {
    'iterations': [100, 200],
    'depth': [3, 10],
    'learning_rate': [0.01, 0.3],
    'l2_leaf_reg': [1, 5],
    'border_count': [32, 128],
    'subsample': [0.8, 1.0],
    'colsample_bylevel': [0.8, 1.0]
}
catboost_cv = KFold(n_splits=5, shuffle=True, random_state=42)
catboost_grid_search = GridSearchCV(catboost_model, catboost_param_grid, cv=catboost_cv,
                                    scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
catboost_grid_search.fit(X_train, y_train)
print("CatBoost Best parameters found: ", catboost_grid_search.best_params_)
catboost_best_model = catboost_grid_search.best_estimator_
catboost_y_pred = catboost_best_model.predict(X_test)

catboost_rmse_test = np.sqrt(mean_squared_error(y_test, catboost_y_pred))
catboost_mae_test = mean_absolute_error(y_test, catboost_y_pred)
catboost_r2_test = r2_score(y_test, catboost_y_pred)
catboost_rpd_test = np.std(y_test) / catboost_rmse_test if catboost_rmse_test != 0 else np.inf
catboost_pearson_test, _ = pearsonr(y_test, catboost_y_pred)

print(f"CatBoost Test RMSE: {catboost_rmse_test:.4f}, R²: {catboost_r2_test:.4f}, MAE: {catboost_mae_test:.4f}")

# ========================
# 6. MLP
# ========================
print("Fitting MLP GridSearchCV...")
mlp_model = MLPRegressor(random_state=42, max_iter=1000, early_stopping=True)
mlp_param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam'],
    'alpha': [0.0001, 0.001],
    'learning_rate_init': [0.001, 0.01]
}
mlp_cv = KFold(n_splits=5, shuffle=True, random_state=42)
mlp_grid_search = GridSearchCV(mlp_model, mlp_param_grid, cv=mlp_cv,
                               scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
mlp_grid_search.fit(X_train, y_train)
print("MLP Best parameters found: ", mlp_grid_search.best_params_)
mlp_best_model = mlp_grid_search.best_estimator_
mlp_y_pred = mlp_best_model.predict(X_test)

mlp_rmse_test = np.sqrt(mean_squared_error(y_test, mlp_y_pred))
mlp_mae_test = mean_absolute_error(y_test, mlp_y_pred)
mlp_r2_test = r2_score(y_test, mlp_y_pred)
mlp_rpd_test = np.std(y_test) / mlp_rmse_test if mlp_rmse_test != 0 else np.inf
mlp_pearson_test, _ = pearsonr(y_test, mlp_y_pred)

print(f"MLP Test RMSE: {mlp_rmse_test:.4f}, R²: {mlp_r2_test:.4f}, MAE: {mlp_mae_test:.4f}")

# ========================
# 7. Stacking Regressor
# ========================
print("Training StackingRegressor...")
base_learners = [
    ("KNN", knn_best_model),
    ("RF", rf_best_model),
    ("XGB", xgb_best_model),
    ("LGBM", lgbm_best_model),
    ("CatBoost", catboost_best_model),
    ("MLP", mlp_best_model)
]
meta_model = LinearRegression()
stacking_regressor = StackingRegressor(
    estimators=base_learners,
    final_estimator=meta_model,
    cv=5,
    n_jobs=-1,
    passthrough=False
)
stacking_regressor.fit(X_train, y_train)

stacking_y_pred = stacking_regressor.predict(X_test)
stacking_rmse_test = np.sqrt(mean_squared_error(y_test, stacking_y_pred))
stacking_mae_test = mean_absolute_error(y_test, stacking_y_pred)
stacking_r2_test = r2_score(y_test, stacking_y_pred)
stacking_rpd_test = np.std(y_test) / stacking_rmse_test if stacking_rmse_test != 0 else np.inf
stacking_pearson_test, _ = pearsonr(y_test, stacking_y_pred)

print(f"Stacking Test RMSE: {stacking_rmse_test:.4f}, R²: {stacking_r2_test:.4f}, MAE: {stacking_mae_test:.4f}")

# ========================
# 8. Plot True vs Predicted graphs for each model (original logic)
# ========================
models_info = [
    ("KNN", y_train, knn_best_model.predict(X_train), y_test, knn_y_pred, r2_score(y_train, knn_best_model.predict(X_train)), knn_r2_test),
    ("RF", y_train, rf_best_model.predict(X_train), y_test, rf_y_pred, r2_score(y_train, rf_best_model.predict(X_train)), rf_r2_test),
    ("XGBoost", y_train, xgb_best_model.predict(X_train), y_test, xgb_y_pred, r2_score(y_train, xgb_best_model.predict(X_train)), xgb_r2_test),
    ("LGBM", y_train, lgbm_best_model.predict(X_train), y_test, lgbm_y_pred, r2_score(y_train, lgbm_best_model.predict(X_train)), lgbm_r2_test),
    ("CatBoost", y_train, catboost_best_model.predict(X_train), y_test, catboost_y_pred, r2_score(y_train, catboost_best_model.predict(X_train)), catboost_r2_test),
    ("MLP", y_train, mlp_best_model.predict(X_train), y_test, mlp_y_pred, r2_score(y_train, mlp_best_model.predict(X_train)), mlp_r2_test),
]

palette = {'Train': '#b4d4e1', 'Test': '#f4ba8a'}
fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=300)
axes = axes.flatten()

for i, (model_name, y_train_true, y_train_pred, y_test_true, y_test_pred, r2_train, r2_test) in enumerate(models_info):
    df_train = pd.DataFrame({'True': y_train_true, 'Predicted': y_train_pred, 'Data Set': 'Train'})
    df_test = pd.DataFrame({'True': y_test_true, 'Predicted': y_test_pred, 'Data Set': 'Test'})
    df_all = pd.concat([df_train, df_test])

    ax = axes[i]
    sns.scatterplot(data=df_all, x="True", y="Predicted", hue="Data Set", palette=palette, alpha=0.5, ax=ax, s=30)
    sns.regplot(data=df_train, x="True", y="Predicted", scatter=False, ax=ax, color=palette['Train'], label='Train Fit')
    sns.regplot(data=df_test, x="True", y="Predicted", scatter=False, ax=ax, color=palette['Test'], label='Test Fit')

    min_val = min(df_all['True'].min(), df_all['Predicted'].min())
    max_val = max(df_all['True'].max(), df_all['Predicted'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.6, linewidth=1)

    ax.set_title(f"{model_name}", fontsize=14, fontweight='bold')
    ax.text(0.05, 0.9, f"Train $R^2$ = {r2_train:.3f}", transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"))
    ax.text(0.05, 0.8, f"Test $R^2$ = {r2_test:.3f}", transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray"))
    ax.set_xlabel("Observed", fontsize=12)
    ax.set_ylabel("Predicted", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("All_Models_True_vs_Predicted.pdf", format='pdf', bbox_inches='tight', dpi=1200)
plt.close()

# ========================
# 9. [New] 5-Fold CV on TRAINING SET for ALL models
# ========================
print("\n" + "="*60)
print("Performing 5-Fold Cross-Validation on TRAINING SET for error bars...")
print("="*60)

scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
models = [
    ("KNN", knn_best_model),
    ("RF", rf_best_model),
    ("XGBoost", xgb_best_model),
    ("LGBM", lgbm_best_model),
    ("CatBoost", catboost_best_model),
    ("MLP", mlp_best_model),
    ("Stacking", stacking_regressor)
]

cv_results = {
    'Model': [],
    'R2_mean': [], 'R2_std': [],
    'RMSE_mean': [], 'RMSE_std': [],
    'MAE_mean': [], 'MAE_std': []
}

for name, model in models:
    print(f"CV for {name}...")
    scores = cross_validate(
        estimator=model,
        X=X_train,
        y=y_train,
        cv=5,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False
    )
    r2 = scores['test_r2']
    rmse = np.sqrt(-scores['test_neg_mean_squared_error'])
    mae = -scores['test_neg_mean_absolute_error']

    cv_results['Model'].append(name)
    cv_results['R2_mean'].append(np.mean(r2))
    cv_results['R2_std'].append(np.std(r2, ddof=1))
    cv_results['RMSE_mean'].append(np.mean(rmse))
    cv_results['RMSE_std'].append(np.std(rmse, ddof=1))
    cv_results['MAE_mean'].append(np.mean(mae))
    cv_results['MAE_std'].append(np.std(mae, ddof=1))

    print(f"  R²: {np.mean(r2):.4f} ± {np.std(r2, ddof=1):.4f}")

cv_df = pd.DataFrame(cv_results)
cv_df.to_excel("cv_results_5fold_on_train.xlsx", index=False)
print("\n✅ CV results saved to 'cv_results_5fold_on_train.xlsx'")

# ========================
# 10. [New] Plot bar charts with error bars (R², RMSE, MAE)
# ========================

# Unified colors (Nature/Science style)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
x = np.arange(len(cv_df))
width = 0.6

# --- R² ---
plt.figure(figsize=(8, 5), dpi=1200)
bars = plt.bar(x, cv_df['R2_mean'], yerr=cv_df['R2_std'], capsize=5,
               color=colors, alpha=0.85, edgecolor='k', linewidth=0.5, width=width)
plt.xticks(x, cv_df['Model'], fontsize=12)
plt.ylabel('$R^2$ (5-Fold CV on Training Set)', fontsize=13)
plt.ylim(0, 1)
plt.title('Model $R^2$ Comparison with Error Bars (±1 SD)', fontsize=14, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, (mean, std) in enumerate(zip(cv_df['R2_mean'], cv_df['R2_std'])):
    plt.text(i, mean + std + 0.01, f'{mean:.3f}', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig("R2_CV_comparison_error_bars.pdf", format='pdf', bbox_inches='tight')
plt.close()

# --- RMSE ---
plt.figure(figsize=(8, 5), dpi=1200)
bars = plt.bar(x, cv_df['RMSE_mean'], yerr=cv_df['RMSE_std'], capsize=5,
               color=colors, alpha=0.85, edgecolor='k', linewidth=0.5, width=width)
plt.xticks(x, cv_df['Model'], fontsize=12)
plt.ylabel('RMSE (5-Fold CV on Training Set)', fontsize=13)
plt.title('Model RMSE Comparison with Error Bars (±1 SD)', fontsize=14, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, (mean, std) in enumerate(zip(cv_df['RMSE_mean'], cv_df['RMSE_std'])):
    plt.text(i, mean + std + 0.01 * mean, f'{mean:.3f}', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig("RMSE_CV_comparison_error_bars.pdf", format='pdf', bbox_inches='tight')
plt.close()

# --- MAE ---
plt.figure(figsize=(8, 5), dpi=1200)
bars = plt.bar(x, cv_df['MAE_mean'], yerr=cv_df['MAE_std'], capsize=5,
               color=colors, alpha=0.85, edgecolor='k', linewidth=0.5, width=width)
plt.xticks(x, cv_df['Model'], fontsize=12)
plt.ylabel('MAE (5-Fold CV on Training Set)', fontsize=13)
plt.title('Model MAE Comparison with Error Bars (±1 SD)', fontsize=14, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, (mean, std) in enumerate(zip(cv_df['MAE_mean'], cv_df['MAE_std'])):
    plt.text(i, mean + std + 0.01 * mean, f'{mean:.3f}', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig("MAE_CV_comparison_error_bars.pdf", format='pdf', bbox_inches='tight')
plt.close()

print("\n✅ All error-bar figures saved as PDF.")

# ========================
# 11. Combined plot for Stacking model (original logic, slightly optimized)
# ========================
# Training set predictions
y_pred_train_stack = stacking_regressor.predict(X_train)

# Evaluation
rmse_train_stack = np.sqrt(mean_squared_error(y_train, y_pred_train_stack))
mae_train_stack = mean_absolute_error(y_train, y_pred_train_stack)
r2_train_stack = r2_score(y_train, y_pred_train_stack)
rmse_test_stack = stacking_rmse_test
mae_test_stack = stacking_mae_test
r2_test_stack = stacking_r2_test

# Confidence interval calculation (simplified version: regression line fit + standard deviation amplification)
scale_factor = 1.5
confidence = 0.95

# Train line
z_train = np.polyfit(y_train, y_pred_train_stack, 1)
p_train = np.poly1d(z_train)
residuals_train = y_pred_train_stack - p_train(y_train)
sigma_train = np.std(residuals_train)
t_val = stats.t.ppf((1 + confidence) / 2., len(y_train) - 2)
x_plot = np.linspace(y_train.min(), y_train.max(), 100)
y_fit_train = p_train(x_plot)
ci_train = t_val * sigma_train * scale_factor * np.sqrt(1/len(y_train) + (x_plot - y_train.mean())**2 / ((y_train - y_train.mean())**2).sum())

# Test line
z_test = np.polyfit(y_test, stacking_y_pred, 1)
p_test = np.poly1d(z_test)
residuals_test = stacking_y_pred - p_test(y_test)
sigma_test = np.std(residuals_test)
y_fit_test = p_test(x_plot)
ci_test = t_val * sigma_test * scale_factor * np.sqrt(1/len(y_test) + (x_plot - y_test.mean())**2 / ((y_test - y_test.mean())**2).sum())

# Plotting
train_color = '#1f77b4'
test_color = '#ff7f0e'
ci_train_color = '#aec7e8'
ci_test_color = '#ffbb78'

fig = plt.figure(figsize=(10, 8), dpi=1200)
gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
ax_main = fig.add_subplot(gs[1:, :-1])
ax_hist_x = fig.add_subplot(gs[0, :-1], sharex=ax_main)
ax_hist_y = fig.add_subplot(gs[1:, -1], sharey=ax_main)

# Main
ax_main.scatter(y_train, y_pred_train_stack, color=train_color, label="Training", alpha=0.6, s=30)
ax_main.plot(x_plot, y_fit_train, color=train_color, lw=1.5,
             label=f"Training Fit\n$R^2$={r2_train_stack:.3f}, MAE={mae_train_stack:.3f}")
ax_main.fill_between(x_plot, y_fit_train - ci_train, y_fit_train + ci_train,
                     color=ci_train_color, alpha=0.4)

ax_main.scatter(y_test, stacking_y_pred, color=test_color, label="Testing", alpha=0.6, s=30)
ax_main.plot(x_plot, y_fit_test, color=test_color, lw=1.5,
             label=f"Testing Fit\n$R^2$={r2_test_stack:.3f}, MAE={mae_test_stack:.3f}")
ax_main.fill_between(x_plot, y_fit_test - ci_test, y_fit_test + ci_test,
                     color=ci_test_color, alpha=0.4)

# 1:1 line
lims = [np.min([y_train.min(), y_test.min(), y_pred_train_stack.min(), stacking_y_pred.min()]),
        np.max([y_train.max(), y_test.max(), y_pred_train_stack.max(), stacking_y_pred.max()])]
ax_main.plot(lims, lims, 'k--', alpha=0.6, label="Ideal Fit", lw=1)
ax_main.set_xlim(lims)
ax_main.set_ylim(lims)

ax_main.set_xlabel("Observed", fontsize=13)
ax_main.set_ylabel("Predicted", fontsize=13)
ax_main.legend(loc="upper left", fontsize=10, framealpha=0.95)

# Histograms
ax_hist_x.hist(y_train, bins=15, color=train_color, alpha=0.7, edgecolor='k', label="Train Obs")
ax_hist_x.hist(y_test, bins=15, color=test_color, alpha=0.7, edgecolor='k', label="Test Obs")
ax_hist_x.tick_params(labelbottom=False)
ax_hist_x.legend(fontsize=9)

ax_hist_y.hist(y_pred_train_stack, bins=15, orientation='horizontal', color=train_color, alpha=0.7, edgecolor='k')
ax_hist_y.hist(stacking_y_pred, bins=15, orientation='horizontal', color=test_color, alpha=0.7, edgecolor='k')
ax_hist_y.set_xlabel("Frequency", fontsize=12)
ax_hist_y.tick_params(labelleft=False)

plt.suptitle("Stacking Regressor: Observed vs Predicted with Confidence Intervals", fontsize=14, fontweight='bold')
plt.savefig('stacking_combined_with_histograms_and_confidence_intervals.pdf', format='pdf', bbox_inches='tight')
plt.show()

print("\n✅ All tasks completed successfully.")