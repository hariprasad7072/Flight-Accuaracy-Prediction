# ==========================================================
# ENHANCED FLIGHT DELAY PREDICTION — TARGET: 90%+ ACCURACY
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_curve, auc, roc_auc_score, precision_recall_curve,
                             f1_score)
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.calibration import CalibratedClassifierCV

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# ----------------------------
# REPRODUCIBILITY
# ----------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ==========================================================
# STEP 1: DATA LOADING
# (Replace this block with your actual data loading code)
# ==========================================================

# Example loader — swap in your actual flight_df loading here:
# flight_df = pd.read_csv("your_flight_data.csv")
#
# Minimum required column: "DELAYED" (binary 0/1 target)

# ---------- SYNTHETIC DATA FOR DEMO (remove if using real data) ----------
from sklearn.datasets import make_classification

X_raw, y_raw = make_classification(
    n_samples=100_000,
    n_features=20,
    n_informative=15,
    n_redundant=3,
    weights=[0.6, 0.4],       # realistic class imbalance
    flip_y=0.02,
    random_state=SEED
)

feature_cols = [f"FEATURE_{i}" for i in range(X_raw.shape[1])]
flight_df = pd.DataFrame(X_raw, columns=feature_cols)
flight_df["DELAYED"] = y_raw
# -------------------------------------------------------------------------

# ==========================================================
# STEP 2: FEATURE ENGINEERING
# ==========================================================

def engineer_features(df):
    """
    Add interaction terms, polynomial features, and domain-aware features.
    Extend this function with real flight columns (hour, carrier, route, etc.)
    """
    df = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "DELAYED"]

    # Interaction features (top-N pairs to avoid explosion)
    top_cols = numeric_cols[:6]
    for i in range(len(top_cols)):
        for j in range(i + 1, len(top_cols)):
            df[f"INTERACT_{top_cols[i]}_{top_cols[j]}"] = (
                df[top_cols[i]] * df[top_cols[j]]
            )

    # Polynomial features on top columns
    for col in top_cols[:4]:
        df[f"{col}_sq"] = df[col] ** 2

    # Rolling statistical aggregates (row-wise std / mean ratio)
    df["ROW_MEAN"]  = df[numeric_cols].mean(axis=1)
    df["ROW_STD"]   = df[numeric_cols].std(axis=1)
    df["ROW_MAX"]   = df[numeric_cols].max(axis=1)
    df["ROW_RANGE"] = df["ROW_MAX"] - df[numeric_cols].min(axis=1)

    return df

flight_df = engineer_features(flight_df)

# ==========================================================
# STEP 3: CLEAN & IMPUTE
# ==========================================================

numeric_df = flight_df.select_dtypes(include=[np.number])

imputer = SimpleImputer(strategy="median")
imputed_array = imputer.fit_transform(numeric_df)
flight_df_clean = pd.DataFrame(imputed_array, columns=numeric_df.columns)

# ==========================================================
# STEP 4: FEATURE / TARGET SPLIT
# ==========================================================

X = flight_df_clean.drop("DELAYED", axis=1)
y = flight_df_clean["DELAYED"].astype(int)

print(f"Dataset shape  : {X.shape}")
print(f"Class balance  : {y.value_counts(normalize=True).to_dict()}")

# ==========================================================
# STEP 5: FEATURE SELECTION (remove low-importance noise)
# ==========================================================

selector_model = RandomForestClassifier(
    n_estimators=100, max_depth=10, random_state=SEED, n_jobs=-1
)
selector_model.fit(X, y)

selector = SelectFromModel(selector_model, threshold="mean", prefit=True)
X_selected = selector.transform(X)

selected_features = X.columns[selector.get_support()]
print(f"Features after selection: {X_selected.shape[1]} / {X.shape[1]}")

# ==========================================================
# STEP 6: SCALE & SPLIT
# ==========================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)

# ==========================================================
# MODEL 1: XGBOOST (tuned)
# ==========================================================

print("\n" + "="*55)
print("MODEL 1 — XGBoost (Tuned)")
print("="*55)

xgb_model = XGBClassifier(
    n_estimators=600,
    learning_rate=0.03,
    max_depth=7,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0.05,
    reg_lambda=1.0,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    random_state=SEED,
    eval_metric="logloss",
    use_label_encoder=False,
    n_jobs=-1
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

xgb_pred = xgb_model.predict(X_test)
xgb_acc  = accuracy_score(y_test, xgb_pred)
print(f"XGBoost Accuracy : {xgb_acc:.4f}")
print(classification_report(y_test, xgb_pred))

# ==========================================================
# MODEL 2: LightGBM (fastest & often top performer)
# ==========================================================

print("\n" + "="*55)
print("MODEL 2 — LightGBM (Tuned)")
print("="*55)

lgbm_model = LGBMClassifier(
    n_estimators=600,
    learning_rate=0.03,
    num_leaves=63,
    max_depth=8,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    class_weight="balanced",
    random_state=SEED,
    n_jobs=-1,
    verbose=-1
)

lgbm_model.fit(X_train, y_train)

lgbm_pred = lgbm_model.predict(X_test)
lgbm_acc  = accuracy_score(y_test, lgbm_pred)
print(f"LightGBM Accuracy : {lgbm_acc:.4f}")
print(classification_report(y_test, lgbm_pred))

# ==========================================================
# MODEL 3: OPTIMIZED RANDOM FOREST
# ==========================================================

print("\n" + "="*55)
print("MODEL 3 — Optimized Random Forest")
print("="*55)

rf_model = RandomForestClassifier(
    n_estimators=400,
    max_depth=20,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=SEED,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_acc  = accuracy_score(y_test, rf_pred)
print(f"Random Forest Accuracy : {rf_acc:.4f}")
print(classification_report(y_test, rf_pred))

# ==========================================================
# MODEL 4: DEEP NEURAL NETWORK (with BatchNorm + EarlyStopping)
# ==========================================================

print("\n" + "="*55)
print("MODEL 4 — Deep Neural Network")
print("="*55)

input_dim = X_train.shape[1]

nn_model = Sequential([
    Dense(256, activation="relu", input_dim=input_dim),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.2),

    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

nn_model.compile(
    loss="binary_crossentropy",
    optimizer=Adam(learning_rate=0.001),
    metrics=["accuracy"]
)

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss"),
    ReduceLROnPlateau(factor=0.5, patience=3, monitor="val_loss")
]

nn_model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=50,
    batch_size=128,
    callbacks=callbacks,
    verbose=1
)

nn_loss, nn_acc = nn_model.evaluate(X_test, y_test, verbose=0)
nn_pred = (nn_model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
print(f"Neural Network Accuracy : {nn_acc:.4f}")
print(classification_report(y_test, nn_pred))

# ==========================================================
# MODEL 5: SOFT VOTING ENSEMBLE (XGB + LGBM + RF)
# ==========================================================

print("\n" + "="*55)
print("MODEL 5 — Soft Voting Ensemble")
print("="*55)

voting_clf = VotingClassifier(
    estimators=[
        ("xgb",  xgb_model),
        ("lgbm", lgbm_model),
        ("rf",   rf_model),
    ],
    voting="soft",
    n_jobs=-1
)

voting_clf.fit(X_train, y_train)

voting_pred = voting_clf.predict(X_test)
voting_acc  = accuracy_score(y_test, voting_pred)
print(f"Voting Ensemble Accuracy : {voting_acc:.4f}")
print(classification_report(y_test, voting_pred))

# ==========================================================
# MODEL 6: STACKING ENSEMBLE (meta-learner = LogisticRegression)
# ==========================================================

print("\n" + "="*55)
print("MODEL 6 — Stacking Ensemble (Meta-Learner)")
print("="*55)

stacking_clf = StackingClassifier(
    estimators=[
        ("xgb",  XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", use_label_encoder=False,
            random_state=SEED, n_jobs=-1
        )),
        ("lgbm", LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            random_state=SEED, n_jobs=-1, verbose=-1
        )),
        ("rf",   RandomForestClassifier(
            n_estimators=200, max_depth=15, random_state=SEED, n_jobs=-1
        )),
    ],
    final_estimator=LogisticRegression(C=1.0, max_iter=1000, random_state=SEED),
    cv=5,
    n_jobs=-1
)

stacking_clf.fit(X_train, y_train)

stacking_pred = stacking_clf.predict(X_test)
stacking_acc  = accuracy_score(y_test, stacking_pred)
print(f"Stacking Ensemble Accuracy : {stacking_acc:.4f}")
print(classification_report(y_test, stacking_pred))

# ==========================================================
# SUMMARY TABLE
# ==========================================================

print("\n" + "="*55)
print("FINAL ACCURACY SUMMARY")
print("="*55)

results = {
    "XGBoost (Tuned)"       : xgb_acc,
    "LightGBM (Tuned)"      : lgbm_acc,
    "Random Forest"         : rf_acc,
    "Neural Network"        : nn_acc,
    "Voting Ensemble"       : voting_acc,
    "Stacking Ensemble"     : stacking_acc,
}

for name, acc in sorted(results.items(), key=lambda x: -x[1]):
    status = "✅ ABOVE 90%" if acc >= 0.90 else "⚠️  Below 90%"
    print(f"  {name:<25} {acc:.4f}  {status}")

best_model_name = max(results, key=results.get)
print(f"\n🏆 Best Model : {best_model_name} ({results[best_model_name]:.4f})")

# ==========================================================
# VISUALISATION: Feature Importance + Accuracy Bar Chart
# ==========================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# --- Feature Importance (XGBoost) ---
importances = xgb_model.feature_importances_
top_n = min(20, len(importances))
sorted_idx = np.argsort(importances)[-top_n:]

axes[0].barh(range(top_n), importances[sorted_idx], color="steelblue")
axes[0].set_yticks(range(top_n))
axes[0].set_yticklabels([f"F{i}" for i in sorted_idx])
axes[0].set_title("XGBoost — Top Feature Importances")
axes[0].set_xlabel("Importance Score")

# --- Accuracy Comparison ---
names  = list(results.keys())
scores = [results[n] for n in names]
colors = ["green" if s >= 0.90 else "salmon" for s in scores]

axes[1].barh(names, scores, color=colors)
axes[1].axvline(0.90, color="red", linestyle="--", linewidth=1.5, label="90% target")
axes[1].set_xlim(0.8, 1.0)
axes[1].set_title("Model Accuracy Comparison")
axes[1].set_xlabel("Accuracy")
axes[1].legend()

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# ==========================================================
# ADDITIONAL EVALUATION GRAPHS: Confusion Matrices
# ==========================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

models_info = [
    ("XGBoost", xgb_pred),
    ("LightGBM", lgbm_pred),
    ("Random Forest", rf_pred),
    ("Neural Network", nn_pred),
    ("Voting Ensemble", voting_pred),
    ("Stacking Ensemble", stacking_pred),
]

for idx, (name, pred) in enumerate(models_info):
    cm = confusion_matrix(y_test, pred)
    axes[idx].imshow(cm, cmap="Blues", aspect="auto")
    axes[idx].set_title(f"{name}\nAccuracy: {accuracy_score(y_test, pred):.4f}")
    axes[idx].set_ylabel("True Label")
    axes[idx].set_xlabel("Predicted Label")
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            axes[idx].text(j, i, str(cm[i, j]), ha="center", va="center", 
                          color="white", fontsize=12, fontweight="bold")
    
    axes[idx].set_xticks([0, 1])
    axes[idx].set_yticks([0, 1])

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()

# ==========================================================
# ROC CURVES & AUC Scores
# ==========================================================

from sklearn.metrics import roc_curve, auc, roc_auc_score

fig, ax = plt.subplots(figsize=(10, 8))

# Get prediction probabilities for each model
xgb_proba   = xgb_model.predict_proba(X_test)[:, 1]
lgbm_proba  = lgbm_model.predict_proba(X_test)[:, 1]
rf_proba    = rf_model.predict_proba(X_test)[:, 1]
nn_proba    = nn_model.predict(X_test, verbose=0).flatten()
voting_proba = voting_clf.predict_proba(X_test)[:, 1]
stacking_proba = stacking_clf.predict_proba(X_test)[:, 1]

model_probas = [
    ("XGBoost", xgb_proba),
    ("LightGBM", lgbm_proba),
    ("Random Forest", rf_proba),
    ("Neural Network", nn_proba),
    ("Voting Ensemble", voting_proba),
    ("Stacking Ensemble", stacking_proba),
]

for name, proba in model_probas:
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc_score = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.4f})", linewidth=2)

ax.plot([0, 1], [0, 1], "k--", label="Random Classifier", linewidth=1)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves Comparison")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()

# ==========================================================
# PRECISION-RECALL CURVES
# ==========================================================

from sklearn.metrics import precision_recall_curve, f1_score

fig, ax = plt.subplots(figsize=(10, 8))

for name, proba in model_probas:
    precision, recall, _ = precision_recall_curve(y_test, proba)
    f1 = f1_score(y_test, (proba > 0.5).astype(int))
    ax.plot(recall, precision, label=f"{name} (F1 = {f1:.4f})", linewidth=2)

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves")
ax.legend(loc="best")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("precision_recall_curves.png", dpi=150, bbox_inches="tight")
plt.close()

# ==========================================================
# SAVE DETAILED CLASSIFICATION REPORTS
# ==========================================================

report_text = "="*70 + "\n"
report_text += "FLIGHT DELAY PREDICTION - COMPREHENSIVE EVALUATION REPORT\n"
report_text += "="*70 + "\n\n"

for name, pred in models_info:
    report_text += f"\n{'='*70}\n"
    report_text += f"MODEL: {name}\n"
    report_text += f"{'='*70}\n"
    report_text += f"Accuracy: {accuracy_score(y_test, pred):.4f}\n"
    report_text += f"\nClassification Report:\n"
    report_text += classification_report(y_test, pred, digits=4)
    report_text += f"\nConfusion Matrix:\n"
    report_text += str(confusion_matrix(y_test, pred)) + "\n"

with open("detailed_evaluation_report.txt", "w") as f:
    f.write(report_text)

# ==========================================================
# SAVE RESULTS SUMMARY AS CSV
# ==========================================================

summary_data = []
for name, pred in models_info:
    proba = None
    for model_name, prob in model_probas:
        if model_name == name:
            proba = prob
            break
    
    acc = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    auc_val = roc_auc_score(y_test, proba)
    
    summary_data.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "AUC-ROC": auc_val,
        "True Negatives": tn,
        "False Positives": fp,
        "False Negatives": fn,
        "True Positives": tp
    })

summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values("Accuracy", ascending=False)
summary_df.to_csv("model_summary_results.csv", index=False)

# ==========================================================
# FEATURE IMPORTANCE COMPARISON (Multiple Models)
# ==========================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# XGBoost Feature Importance
xgb_importances = xgb_model.feature_importances_
top_n = min(15, len(xgb_importances))
sorted_idx = np.argsort(xgb_importances)[-top_n:]

axes[0].barh(range(top_n), xgb_importances[sorted_idx], color="steelblue")
axes[0].set_yticks(range(top_n))
axes[0].set_yticklabels([f"Feature {i}" for i in sorted_idx])
axes[0].set_title("XGBoost — Top 15 Feature Importances")
axes[0].set_xlabel("Importance Score")

# LightGBM Feature Importance
lgbm_importances = lgbm_model.feature_importances_
sorted_idx_lgbm = np.argsort(lgbm_importances)[-top_n:]

axes[1].barh(range(top_n), lgbm_importances[sorted_idx_lgbm], color="darkorange")
axes[1].set_yticks(range(top_n))
axes[1].set_yticklabels([f"Feature {i}" for i in sorted_idx_lgbm])
axes[1].set_title("LightGBM — Top 15 Feature Importances")
axes[1].set_xlabel("Importance Score")

plt.tight_layout()
plt.savefig("feature_importance_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# ==========================================================
# MODEL METRICS RADAR CHART
# ==========================================================

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection="polar")

metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]

colors_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

for idx, (name, pred) in enumerate(models_info):
    proba = [p for m, p in model_probas if m == name][0]
    
    acc = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    values = [acc, precision, recall, f1]
    values += values[:1]
    
    ax.plot(angles, values, "o-", linewidth=2, label=name, color=colors_list[idx])
    ax.fill(angles, values, alpha=0.15, color=colors_list[idx])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1)
ax.set_title("Model Performance Comparison (Radar Chart)", size=14, weight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
ax.grid(True)

plt.tight_layout()
plt.savefig("model_performance_radar.png", dpi=150, bbox_inches="tight")
plt.close()

# ==========================================================
# PRINT SUMMARY TABLE
# ==========================================================

print("\n" + "="*70)
print("COMPREHENSIVE MODEL EVALUATION SUMMARY (sorted by Accuracy)")
print("="*70)
print(summary_df.to_string(index=False))

# ==========================================================
# 5-FOLD CROSS-VALIDATION ANALYSIS
# ==========================================================

print("\n" + "="*70)
print("5-FOLD CROSS-VALIDATION RESULTS")
print("="*70)

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

cv_results = {}
model_list = [
    ("XGBoost", xgb_model),
    ("LightGBM", lgbm_model),
    ("Random Forest", rf_model),
    ("Voting Ensemble", voting_clf),
    ("Stacking Ensemble", stacking_clf),
]

for name, model in model_list:
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring="accuracy", n_jobs=-1)
    cv_results[name] = {
        "mean": cv_scores.mean(),
        "std": cv_scores.std(),
        "scores": cv_scores,
        "min": cv_scores.min(),
        "max": cv_scores.max()
    }
    print(f"\n{name}:")
    print(f"  Mean Accuracy    : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"  Fold Scores      : {', '.join([f'{s:.4f}' for s in cv_scores])}")
    print(f"  Min/Max          : {cv_scores.min():.4f} / {cv_scores.max():.4f}")

# ==========================================================
# CROSS-VALIDATION SCORES VISUALIZATION
# ==========================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Box plot of CV scores
cv_names = list(cv_results.keys())
cv_data = [cv_results[name]["scores"] for name in cv_names]

ax1.boxplot(cv_data, labels=cv_names, patch_artist=True, 
            boxprops=dict(facecolor='lightblue', alpha=0.7),
            medianprops=dict(color='red', linewidth=2))
ax1.axhline(0.90, color='green', linestyle='--', linewidth=2, label='90% Target')
ax1.set_ylabel('Accuracy Score')
ax1.set_title('5-Fold Cross-Validation Score Distribution')
ax1.legend()
ax1.grid(alpha=0.3)
ax1.set_ylim(0.88, 1.02)

# Mean CV scores with error bars
means = [cv_results[name]["mean"] for name in cv_names]
stds = [cv_results[name]["std"] for name in cv_names]
colors = ['green' if m >= 0.90 else 'salmon' for m in means]

ax2.bar(range(len(cv_names)), means, yerr=stds, capsize=5, color=colors, alpha=0.7)
ax2.axhline(0.90, color='red', linestyle='--', linewidth=2, label='90% Target')
ax2.set_xticks(range(len(cv_names)))
ax2.set_xticklabels(cv_names, rotation=45, ha='right')
ax2.set_ylabel('Mean Accuracy')
ax2.set_title('Cross-Validation Mean Scores with Std Dev')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim(0.88, 1.02)

plt.tight_layout()
plt.savefig("cross_validation_analysis.png", dpi=150, bbox_inches="tight")
plt.close()

# ==========================================================
# LEARNING CURVES FOR KEY MODELS
# ==========================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

key_models = [
    ("XGBoost", xgb_model),
    ("LightGBM", lgbm_model),
    ("Neural Network", nn_model if hasattr(nn_model, 'predict') else None),
    ("Random Forest", rf_model),
]

train_sizes = np.linspace(0.1, 1.0, 10)

for idx, (name, model) in enumerate([m for m in key_models if m[1] is not None]):
    if name == "Neural Network":
        # Skip neural network learning curve for computational reasons
        axes[idx].text(0.5, 0.5, f'{name}:\nLearning curve skipped for NN\n(use validation metrics instead)',
                       ha='center', va='center', fontsize=14, transform=axes[idx].transAxes)
        axes[idx].set_title(f'{name} - Learning Curve')
        continue
    
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X_train, y_train, cv=5, 
        train_sizes=np.linspace(0.05, 1.0, 10),
        scoring='accuracy', n_jobs=-1, verbose=0
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    axes[idx].plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training Score', linewidth=2)
    axes[idx].fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, 
                           color='blue', alpha=0.2)
    axes[idx].plot(train_sizes_abs, val_mean, 'o-', color='red', label='Validation Score', linewidth=2)
    axes[idx].fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, 
                           color='red', alpha=0.2)
    axes[idx].axhline(0.90, color='green', linestyle='--', linewidth=1.5, label='90% Target')
    
    axes[idx].set_xlabel('Training Set Size')
    axes[idx].set_ylabel('Accuracy Score')
    axes[idx].set_title(f'{name} - Learning Curve')
    axes[idx].legend(loc='best')
    axes[idx].grid(alpha=0.3)
    axes[idx].set_ylim(0.85, 1.02)

plt.tight_layout()
plt.savefig("learning_curves.png", dpi=150, bbox_inches="tight")
plt.close()

# ==========================================================
# CALIBRATION CURVES
# ==========================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

calibration_models = [
    ("XGBoost", xgb_proba),
    ("LightGBM", lgbm_proba),
    ("Random Forest", rf_proba),
    ("Neural Network", nn_proba),
    ("Voting Ensemble", voting_proba),
    ("Stacking Ensemble", stacking_proba),
]

for idx, (name, proba) in enumerate(calibration_models):
    prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10, strategy='uniform')
    
    axes[idx].plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated', linewidth=2)
    axes[idx].plot(prob_pred, prob_true, 'o-', label='Model Calibration', linewidth=2, markersize=8)
    axes[idx].fill_between(prob_pred, prob_true, prob_pred, alpha=0.2)
    
    axes[idx].set_xlabel('Mean Predicted Probability')
    axes[idx].set_ylabel('Fraction of Positives')
    axes[idx].set_title(f'{name}\nCalibration Curve')
    axes[idx].legend(loc='best')
    axes[idx].set_xlim(0, 1)
    axes[idx].set_ylim(0, 1)
    axes[idx].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("calibration_curves.png", dpi=150, bbox_inches="tight")
plt.close()

# ==========================================================
# CUMULATIVE GAIN CHART
# ==========================================================

fig, ax = plt.subplots(figsize=(12, 8))

for name, proba in model_probas:
    # Sort by predicted probability in descending order
    sorted_indices = np.argsort(proba)[::-1]
    y_sorted = y_test.values[sorted_indices]
    
    # Calculate cumulative positives (gains)
    n_positives = (y_test == 1).sum()
    cumulative_positives = np.cumsum(y_sorted) / n_positives
    percentile = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
    
    ax.plot(percentile, cumulative_positives, label=f'{name}', linewidth=2.5)

# Baseline (random)
ax.plot([0, 1], [0, 1], 'k--', label='Random Model', linewidth=2)

# Perfect model
n_positives = (y_test == 1).sum()
ax.plot([0, n_positives/len(y_test), 1], [0, 1, 1], 'g--', label='Perfect Model', linewidth=2)

ax.set_xlabel('Percentage of Dataset Targeted (%)', fontsize=12)
ax.set_ylabel('Cumulative Gain (% of Positives Found)', fontsize=12)
ax.set_title('Cumulative Gain Chart — Model Ranking Ability', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=11)
ax.grid(alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig("cumulative_gain_chart.png", dpi=150, bbox_inches="tight")
plt.close()

# ==========================================================
# LIFT CHART
# ==========================================================

fig, ax = plt.subplots(figsize=(12, 8))

for name, proba in model_probas:
    sorted_indices = np.argsort(proba)[::-1]
    y_sorted = y_test.values[sorted_indices]
    
    n_positives = (y_test == 1).sum()
    baseline_rate = n_positives / len(y_test)
    
    # Calculate lift at different percentiles
    percentiles = np.linspace(0.01, 1.0, 100)
    lifts = []
    
    for perc in percentiles:
        idx = int(len(y_sorted) * perc)
        if idx > 0:
            hit_rate = np.mean(y_sorted[:idx])
            lift = hit_rate / baseline_rate if baseline_rate > 0 else 1
            lifts.append(lift)
        else:
            lifts.append(1)
    
    ax.plot(percentiles, lifts, label=f'{name}', linewidth=2.5)

ax.axhline(1.0, color='k', linestyle='--', linewidth=2, label='Baseline (Lift=1)')
ax.set_xlabel('Percentage of Dataset Targeted (%)', fontsize=12)
ax.set_ylabel('Lift', fontsize=12)
ax.set_title('Lift Chart — Relative Model Performance', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=11)
ax.grid(alpha=0.3)
ax.set_xlim(0, 1)

plt.tight_layout()
plt.savefig("lift_chart.png", dpi=150, bbox_inches="tight")
plt.close()

# ==========================================================
# MODEL RELIABILITY DIAGRAM
# ==========================================================

fig, ax = plt.subplots(figsize=(14, 8))

x_pos = np.arange(len(cv_names))
width = 0.35

mean_scores = [cv_results[name]["mean"] for name in cv_names]
test_scores = [results[name] for name in cv_names]

bars1 = ax.bar(x_pos - width/2, mean_scores, width, label='CV Mean Score', alpha=0.8, color='steelblue')
bars2 = ax.bar(x_pos + width/2, test_scores, width, label='Test Set Score', alpha=0.8, color='coral')

ax.axhline(0.90, color='green', linestyle='--', linewidth=2, label='90% Target', alpha=0.7)
ax.set_ylabel('Accuracy Score', fontsize=12)
ax.set_title('Cross-Validation vs Test Set Performance\n(Model Generalization Ability)', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(cv_names, rotation=45, ha='right')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0.92, 1.0)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig("model_reliability_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# ==========================================================
# THRESHOLD OPTIMIZATION ANALYSIS
# ==========================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Use best model (Neural Network)
best_proba = nn_proba
thresholds = np.linspace(0, 1, 101)

precisions = []
recalls = []
f1_scores = []
accuracies = []

for threshold in thresholds:
    pred = (best_proba > threshold).astype(int)
    precisions.append(np.mean(y_test[pred == 1]) if (pred == 1).sum() > 0 else 0)
    recalls.append(np.sum((pred == 1) & (y_test == 1)) / np.sum(y_test == 1) if np.sum(y_test == 1) > 0 else 0)
    f1_scores.append(f1_score(y_test, pred) if len(np.unique(pred)) > 1 else 0)
    accuracies.append(accuracy_score(y_test, pred))

axes[0, 0].plot(thresholds, precisions, linewidth=2.5, color='blue')
axes[0, 0].set_xlabel('Classification Threshold')
axes[0, 0].set_ylabel('Precision')
axes[0, 0].set_title('Precision vs Threshold (Neural Network)')
axes[0, 0].grid(alpha=0.3)
axes[0, 0].axvline(0.5, color='red', linestyle='--', alpha=0.7, label='Default (0.5)')
axes[0, 0].legend()

axes[0, 1].plot(thresholds, recalls, linewidth=2.5, color='green')
axes[0, 1].set_xlabel('Classification Threshold')
axes[0, 1].set_ylabel('Recall')
axes[0, 1].set_title('Recall vs Threshold (Neural Network)')
axes[0, 1].grid(alpha=0.3)
axes[0, 1].axvline(0.5, color='red', linestyle='--', alpha=0.7, label='Default (0.5)')
axes[0, 1].legend()

axes[1, 0].plot(thresholds, f1_scores, linewidth=2.5, color='purple')
max_f1_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[max_f1_idx]
axes[1, 0].axvline(optimal_threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Optimal ({optimal_threshold:.2f})')
axes[1, 0].set_xlabel('Classification Threshold')
axes[1, 0].set_ylabel('F1-Score')
axes[1, 0].set_title('F1-Score vs Threshold (Neural Network)')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

axes[1, 1].plot(thresholds, accuracies, linewidth=2.5, color='orange')
axes[1, 1].set_xlabel('Classification Threshold')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].set_title('Accuracy vs Threshold (Neural Network)')
axes[1, 1].grid(alpha=0.3)
axes[1, 1].axvline(0.5, color='red', linestyle='--', alpha=0.7, label='Default (0.5)')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig("threshold_optimization.png", dpi=150, bbox_inches="tight")
plt.close()

# ==========================================================
# MODEL COMPARISON HEATMAP (Metrics)
# ==========================================================

metrics_heatmap = summary_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']].set_index('Model')

fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(metrics_heatmap.values, cmap='RdYlGn', aspect='auto', vmin=0.9, vmax=1.0)

ax.set_xticks(np.arange(len(metrics_heatmap.columns)))
ax.set_yticks(np.arange(len(metrics_heatmap.index)))
ax.set_xticklabels(metrics_heatmap.columns, fontsize=11, fontweight='bold')
ax.set_yticklabels(metrics_heatmap.index, fontsize=11)

# Add text annotations
for i in range(len(metrics_heatmap.index)):
    for j in range(len(metrics_heatmap.columns)):
        text = ax.text(j, i, f'{metrics_heatmap.values[i, j]:.4f}',
                      ha="center", va="center", color="black", fontweight='bold', fontsize=10)

ax.set_title('Model Performance Metrics Heatmap', fontsize=14, fontweight='bold', pad=20)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Score', rotation=270, labelpad=20)

plt.tight_layout()
plt.savefig("metrics_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n" + "="*70)
print("✅  PROJECT EXECUTION COMPLETED SUCCESSFULLY")
print("="*70)
print("\n📊  COMPREHENSIVE ANALYTICS GENERATED:")
print("   • Cross-Validation Analysis        — 5-fold CV scores with distribution")
print("   • Learning Curves                  — Training vs validation curves")
print("   • Calibration Curves               — Probability calibration analysis")
print("   • Cumulative Gain Chart            — Model ranking ability")
print("   • Lift Chart                       — Relative model performance")
print("   • Model Reliability Comparison     — Generalization ability")
print("   • Threshold Optimization           — Optimal decision thresholds")
print("   • Metrics Heatmap                  — Performance metrics comparison")
print("   • Plus all previous 8 graphs!")
print("\n" + "="*70)