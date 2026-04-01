import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Ensure folder
os.makedirs("models", exist_ok=True)

# Data (2 features recommended)
X = [
    [10, 25],
    [20, 30],
    [30, 35],
    [40, 40]
]

y = [100, 200, 300, 400]

# ✅ Step 1: Polynomial
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# ✅ Step 2: Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

# ✅ Step 3: Model
model = RandomForestRegressor()
model.fit(X_scaled, y)

# Save paths
model_path = "models/best_model.pkl"
scaler_path = "models/scaler.pkl"
poly_path = "models/poly.pkl"

# ✅ Save ALL
joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(poly, poly_path)

# Verify
print("Model size:", os.path.getsize(model_path))
print("Scaler size:", os.path.getsize(scaler_path))
print("Poly size:", os.path.getsize(poly_path))

import joblib

poly = joblib.load('models/poly.pkl')
scaler = joblib.load('models/scaler.pkl')
model = joblib.load('models/best_model.pkl')

print("All files saved successfully!")
