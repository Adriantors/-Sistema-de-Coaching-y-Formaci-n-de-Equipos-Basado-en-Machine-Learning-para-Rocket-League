"""
Rank Prediction Model Training

This module implements the training process for the Rocket League rank prediction model.
It uses LightGBM for model training and includes hyperparameter optimization with Optuna.

Author: Adrián Torremocha
Date: 02-05-2025
"""

import warnings, optuna, joblib, shap
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from typing import Dict, List, Tuple, Optional

###############################################################################
# 1. CONFIGURACIÓN
###############################################################################
CSV_PATH   = Path("data/players_anonymous_per_game.csv")
OUTPUT_DIR = Path("outputs"); OUTPUT_DIR.mkdir(exist_ok=True)
SEED       = 727
N_TRIALS   = 50         
FOLDS      = 5
TARGET     = "rank_id"

###############################################################################
# 2. UTILIDADES
###############################################################################

def safe_divide(a, b, fill_value=0):
    """Divide a/b evitando inf y NaN."""
    with np.errstate(divide="ignore", invalid="ignore"):
        res = np.divide(a, b)
    if isinstance(res, (pd.Series, pd.DataFrame)):
        return res.replace([np.inf, -np.inf], fill_value).fillna(fill_value)
    return fill_value if np.isinf(res) or np.isnan(res) else res

###############################################################################
# 3. CARGA Y FEATURE ENGINEERING
###############################################################################
print("Leyendo", CSV_PATH)
df = pd.read_csv(CSV_PATH, sep=";")

rank_order = [
    "Bronce", "Plata", "Oro", "Platino", "Diamante",
    "Champion", "Grand Champion", "Supersonic Legend",
]
enc = OrdinalEncoder(categories=[rank_order])
df = df[df["Rank"].isin(rank_order)]
df[TARGET] = enc.fit_transform(df[["Rank"]]).astype(int)

a = df  # alias solo para compactar
print("Derivando métricas…")

# Eficiencia de tiro / parada
a["shot_accuracy"] = safe_divide(a["goals per game"],  a["shots per game"])
a["save_ratio"]    = safe_divide(a["saves per game"],  a["shots conceded per game"])

# Estadísticas de boost
a["boost_collection_efficiency"] = safe_divide(a["amount collected small pads per game"], a["amount collected per game"])
a["boost_steal_ratio"]           = safe_divide(a["amount stolen per game"],             a["amount collected per game"])

# Posicionamiento y movimiento
air = a["time low in air per game"] + a["time high in air per game"] + 1
a["ground_air_ratio"]      = safe_divide(a["time on ground per game"], air)
a["offensive_positioning"] = safe_divide(a["time offensive third per game"], a["time defensive third per game"], 1)
a["ball_control"]          = safe_divide(a["time in front of ball per game"], a["time behind ball per game"], 1)
a["speed_management"]      = safe_divide(a["time supersonic speed per game"], a["time boost speed per game"], 1)

# Rotación / equipo
a["rotation_balance"]   = safe_divide(a["time most back per game"], a["time most forward per game"], 1)
a["defensive_presence"] = safe_divide(a["time defensive half per game"], a["time offensive half per game"], 1)
a["teammate_spacing"]   = safe_divide(a["avg distance to team mates per game"], a["avg distance to ball per game"], 1)

# Agresividad
min_demos = a["demos taken per game"].replace(0, 1)
a["demo_efficiency"] = safe_divide(a["demos inflicted per game"], min_demos)

a["boost_denial"] = safe_divide(
    a["amount stolen big pads per game"] + a["amount stolen small pads per game"],
    a["amount collected per game"],
)

# Totales sencillos
a["total_off_actions"] = a["goals per game"] + a["assists per game"] + a["shots per game"]
a["total_def_actions"] = a["saves per game"] + a["demos taken per game"]

a.columns = [c.replace(" ", "_") for c in a.columns]
num_cols  = a.select_dtypes("number").columns
a[num_cols] = a[num_cols].replace([np.inf, -np.inf], np.nan).fillna(a[num_cols].mean())

X = a.select_dtypes("number").drop(columns=[TARGET])
y = a[TARGET]

###############################################################################
# 4. SETUP DE OPTUNA (GBDT REGRESSION)
###############################################################################
base_params = dict(
    objective="regression",
    boosting_type="gbdt",
    metric="rmse",
    class_weight="balanced",
    seed=SEED,
    verbose=-1,
)

skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)


def qwk(pred, truth):
    return cohen_kappa_score(truth, np.clip(np.rint(pred), 0, len(rank_order)-1), weights="quadratic")


def objective(trial):
    params = base_params | {
        "learning_rate":     trial.suggest_float("lr",     0.04, 0.12, log=True),
        "num_leaves":        trial.suggest_int("leaves",  64, 200),
        "max_depth":         trial.suggest_int("depth",    6, 11),
        "min_data_in_leaf":  trial.suggest_int("min_data", 30, 150),
        "feature_fraction":  trial.suggest_float("feat_f", 0.7, 1.0),
        "bagging_fraction":  trial.suggest_float("bag_f",  0.7, 1.0),
        "lambda_l1":         trial.suggest_float("l1",     1e-8, 0.5, log=True),
        "lambda_l2":         trial.suggest_float("l2",     1e-8, 0.5, log=True),
    }
    folds = []
    for tr_idx, va_idx in skf.split(X, y):
        m = lgb.train(params, lgb.Dataset(X.iloc[tr_idx], y.iloc[tr_idx]), 800,
                      valid_sets=[lgb.Dataset(X.iloc[va_idx], y.iloc[va_idx])],
                      callbacks=[lgb.early_stopping(60, verbose=False)])
        folds.append(qwk(m.predict(X.iloc[va_idx]), y.iloc[va_idx]))
    return np.mean(folds)

print("Buscando hiper‑parámetros … (30 trials)")
warnings.filterwarnings("ignore", category=UserWarning)
study = optuna.create_study(direction="maximize", study_name="gbdt_qwk")
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

best = study.best_params
print("Mejores params →", best)
final_params = base_params | best

###############################################################################
# 5. ENSEMBLE DE 5 SEMILLAS
###############################################################################
print("Entrenando ensemble …")
models, qwk_all = [], []
for seed in range(SEED, SEED+5):
    final_params["seed"] = seed
    fold_qwk = []
    for tr_idx, va_idx in skf.split(X, y):
        m = lgb.train(final_params, lgb.Dataset(X.iloc[tr_idx], y.iloc[tr_idx]), 800,
                      valid_sets=[lgb.Dataset(X.iloc[va_idx], y.iloc[va_idx])],
                      callbacks=[lgb.early_stopping(60, verbose=False)])
        fold_qwk.append(qwk(m.predict(X.iloc[va_idx]), y.iloc[va_idx]))
    print(f"Seed {seed}: QWK {np.mean(fold_qwk):.4f}")
    qwk_all.extend(fold_qwk)
    models.append(m)
print(f"Media ensemble: {np.mean(qwk_all):.4f} ± {np.std(qwk_all):.4f}")

###############################################################################
# 6. GUARDAR ARTEFACTOS
###############################################################################
for i, mdl in enumerate(models):
    mdl.save_model(OUTPUT_DIR / f"rank_predictor_gbdt_{i}.txt")

rank_stats = X.assign(rank=y).groupby("rank").agg(["mean", "std"])
rank_stats.to_parquet(OUTPUT_DIR / "rank_stats.parquet")
shap_exp = shap.TreeExplainer(models[0])
joblib.dump(shap_exp, OUTPUT_DIR / "shap_explainer.pkl")
print("Artefactos almacenados en", OUTPUT_DIR)
