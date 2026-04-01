import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# ====================== Load Data ======================
X_train = np.load('../notebooks/X_train_scaled.npy')
X_test = np.load('../notebooks/X_test_scaled.npy')
y_train = np.load('../notebooks/y_train.npy')
y_test = np.load('../notebooks/y_test.npy')

# ====================== Optimized Base Models ======================

# Faster RandomForest
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    n_jobs=-1,
    random_state=42
)

# Fast & tuned XGBoost (use histogram for speed)
xgb = XGBRegressor(
    tree_method='hist',     # ⚡ faster
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# LightGBM (very fast)
lgb = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42
)

# ====================== Fast Hyperparameter Tuning ======================
param_dist = {
    'n_estimators': [200, 300, 400],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.03, 0.05, 0.1]
}

random_search = RandomizedSearchCV(
    xgb,
    param_distributions=param_dist,
    n_iter=10,              # ⚡ faster than GridSearch
    cv=3,
    scoring='r2',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

random_search.fit(X_train, y_train)
best_xgb = random_search.best_estimator_

# ====================== Stacking Model ======================
estimators = [
    ('rf', rf),
    ('xgb', best_xgb),
    ('lgb', lgb)
]

stack = StackingRegressor(
    estimators=estimators,
    final_estimator=LGBMRegressor(),  # ⚡ faster than RF
    n_jobs=-1,
    passthrough=True
)

stack.fit(X_train, y_train)

# ====================== Evaluation ======================
pred = stack.predict(X_test)

print("\n🎯 FINAL MODEL PERFORMANCE")
print(f"R2 Score  : {r2_score(y_test, pred):.4f}")
print(f"MAE       : {mean_absolute_error(y_test, pred):.4f}")
print(f"RMSE      : {np.sqrt(mean_squared_error(y_test, pred)):.4f}")

# ====================== Save Model ======================
joblib.dump(stack, '../models/best_model.pkl')
print("\n✅ Best Model Saved Successfully!")