import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None


DATA_PATH = Path("data/crop_yield_data.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path: Path):
    df = pd.read_csv(path)
    if "yeild" in df.columns:
        df = df.rename(columns={"yeild": "Yield"})
    elif "yield" in df.columns:
        df = df.rename(columns={"yield": "Yield"})

    df = df.dropna().drop_duplicates().reset_index(drop=True)
    X = df[["Fertilizer", "temp", "N", "P", "K"]]
    y = df["Yield"]
    return X, y


def make_pipeline(model):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def evaluate_model(pipe, X, y):
    scores = cross_val_score(pipe, X.values, y.values, cv=5, scoring="r2", n_jobs=-1)
    return np.mean(scores), np.std(scores)


def tune_best_model(name, pipeline, X, y):
    if name == "random_forest":
        param_dist = {
            "model__n_estimators": [200, 300, 400, 500],
            "model__max_depth": [8, 10, 12, None],
            "model__min_samples_split": [2, 4, 6],
            "model__min_samples_leaf": [1, 2, 4],
        }
    elif name == "lightgbm":
        param_dist = {
            "model__n_estimators": [200, 300, 400, 500],
            "model__learning_rate": [0.01, 0.03, 0.05],
            "model__num_leaves": [31, 50, 70],
            "model__max_depth": [6, 8, 10, -1],
        }
    elif name == "xgboost":
        param_dist = {
            "model__n_estimators": [200, 300, 400],
            "model__learning_rate": [0.01, 0.03, 0.05],
            "model__max_depth": [4, 6, 8],
            "model__subsample": [0.7, 0.8, 0.9],
            "model__colsample_bytree": [0.7, 0.8, 0.9],
        }
    else:
        return pipeline

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=20,
        cv=3,
        scoring="r2",
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )
    search.fit(X.values, y.values)
    print(f"\nTuned {name} with best params: {search.best_params_}")
    return search.best_estimator_


def main():
    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    candidates = {}

    if LGBMRegressor is not None:
        candidates["lightgbm"] = make_pipeline(
            LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42,
                n_jobs=-1,
            )
        )

    if XGBRegressor is not None:
        candidates["xgboost"] = make_pipeline(
            XGBRegressor(
                tree_method="hist",
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )
        )

    if CatBoostRegressor is not None:
        candidates["catboost"] = make_pipeline(
            CatBoostRegressor(
                iterations=300,
                learning_rate=0.05,
                depth=6,
                verbose=0,
                random_state=42,
            )
        )

    candidates["random_forest"] = make_pipeline(
        RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            n_jobs=-1,
            random_state=42,
        )
    )

    results = []
    print("Evaluating candidate models with 5-fold cross-validation...\n")
    for name, pipe in candidates.items():
        mean_r2, std_r2 = evaluate_model(pipe, X_train, y_train)
        results.append((name, mean_r2, std_r2, pipe))
        print(f"{name:12s}  R2 mean: {mean_r2:.4f}  std: {std_r2:.4f}")

    results.sort(key=lambda item: item[1], reverse=True)
    best_name, best_r2, best_std, best_pipeline = results[0]

    print(f"\nBest model: {best_name} (CV R2 = {best_r2:.4f} ± {best_std:.4f})")

    print(f"\nTuning the best model ({best_name}) for final performance...")
    best_pipeline = tune_best_model(best_name, best_pipeline, X_train, y_train)

    print("\nFitting the best tuned model on full training data...")
    best_pipeline.fit(X_train.values, y_train.values)
    y_pred = best_pipeline.predict(X_test.values)

    print("\nTest set performance")
    print(f"R2   : {r2_score(y_test, y_pred):.4f}")
    print(f"MAE  : {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"RMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

    model_path = MODEL_DIR / "best_pipeline.pkl"
    joblib.dump(best_pipeline, model_path)
    meta = {
        "model": best_name,
        "cv_r2_mean": float(best_r2),
        "cv_r2_std": float(best_std),
        "test_r2": float(r2_score(y_test, y_pred)),
        "test_mae": float(mean_absolute_error(y_test, y_pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "features": ["Fertilizer", "temp", "N", "P", "K"],
    }
    (MODEL_DIR / "best_model_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved best pipeline to: {model_path}")


if __name__ == "__main__":
    main()
