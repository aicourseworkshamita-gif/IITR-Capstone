import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.shipment_prep import FEATURES

def run_part_b():
    df = pd.read_csv("data/modelling_table.csv")
    X = df[FEATURES]
    y = df['delivery_hours']
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=180/1200, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=180/1020, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    pd.DataFrame([
        ("train_rows", len(X_train)), ("val_rows", len(X_val)), ("test_rows", len(X_test)),
        ("train_scaled_max_abs_mean", float(np.abs(X_train_scaled.mean(axis=0)).max())),
        ("test_scaled_max_abs_mean", float(np.abs(X_test_scaled.mean(axis=0)).max()))
    ], columns=['metric', 'value']).to_csv("outputs/part_b_checks.csv", index=False)
    
    val_pred_mean = np.full_like(y_val, y_train.mean(), dtype=float)
    metrics_rows = [{
        "model": "mean_baseline", "split": "val",
        "MAE": round(mean_absolute_error(y_val, val_pred_mean), 4),
        "MSE": round(mean_squared_error(y_val, val_pred_mean), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_val, val_pred_mean)), 4),
        "R2": round(r2_score(y_val, val_pred_mean), 4)
    }]
    
    depths = [1, 2, 3, 4, 6, 8, 12, None]
    depth_rows = []
    tree_results = {}
    for d in depths:
        dt = DecisionTreeRegressor(max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        va_rmse = np.sqrt(mean_squared_error(y_val, dt.predict(X_val)))
        depth_rows.append({"max_depth": str(d) if d is not None else "unlimited", "train_rmse": round(np.sqrt(mean_squared_error(y_train, dt.predict(X_train))), 4), "val_rmse": round(va_rmse, 4)})
        tree_results[d] = va_rmse
    pd.DataFrame(depth_rows).to_csv("outputs/part_b_tree_depth.csv", index=False)
    best_depth = min(tree_results, key=lambda k: (tree_results[k], 99 if k is None else k))
    
    models = {
        "linear_regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    for name, m in models.items():
        m.fit(X_train, y_train)
        pred = m.predict(X_val)
        metrics_rows.append({
            "model": name, "split": "val",
            "MAE": round(mean_absolute_error(y_val, pred), 4),
            "MSE": round(mean_squared_error(y_val, pred), 4),
            "RMSE": round(np.sqrt(mean_squared_error(y_val, pred)), 4),
            "R2": round(r2_score(y_val, pred), 4)
        })
    
    lr_exp = []
    for lr in [0.01, 0.1, 1.0]:
        gb_lr = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42)
        gb_lr.fit(X_train, y_train)
        lr_exp.append({"learning_rate": lr, "val_rmse": round(np.sqrt(mean_squared_error(y_val, gb_lr.predict(X_val))), 4), "val_r2": round(r2_score(y_val, gb_lr.predict(X_val)), 4)})
    pd.DataFrame(lr_exp).to_csv("outputs/part_b_gb_learning_rate.csv", index=False)
    
    cand = sorted([m for m in metrics_rows if m["model"] != "mean_baseline"], key=lambda x: x["RMSE"])
    best_name, second_name = cand[0]["model"], cand[1]["model"]
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    X_tv, y_tv = pd.concat([X_train, X_val]), pd.concat([y_train, y_val])
    cv_rows = []
    for name in [best_name, second_name]:
        scores = cross_val_score(models[name], X_tv, y_tv, cv=kf, scoring='r2')
        cv_rows.append({"model": name, "mean_r2": round(scores.mean(), 4), "std_r2": round(scores.std(), 4)})
    pd.DataFrame(cv_rows).to_csv("outputs/part_b_cv.csv", index=False)
    
    final_model = models[best_name]
    final_model.fit(X_train, y_train)
    test_pred = final_model.predict(X_test)
    metrics_rows.append({
        "model": best_name, "split": "test",
        "MAE": round(mean_absolute_error(y_test, test_pred), 4),
        "MSE": round(mean_squared_error(y_test, test_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, test_pred)), 4),
        "R2": round(r2_score(y_test, test_pred), 4)
    })
    pd.DataFrame(metrics_rows).to_csv("outputs/part_b_metrics.csv", index=False)
    print("Part B completed successfully.")

if __name__ == "__main__":
    run_part_b()
