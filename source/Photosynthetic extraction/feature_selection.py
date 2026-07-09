import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from boruta import BorutaPy

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.svm import SVR

from statsmodels.stats.outliers_influence import variance_inflation_factor

import shap

# =====================================================
# path
# =====================================================

path = r"C:\Users\Jeron\Desktop\code\反演\data\光谱+辐射+纹理+表型反演光合参数数据集.xlsx"

sheet = "Sheet1"

# =====================================================
# read
# =====================================================

df = pd.read_excel(
    path,
    sheet_name=sheet
)

# =====================================================
# target
# =====================================================

target = "Pn"

# =====================================================
# feature columns
# =====================================================

id_col = "ID"

feature_cols = [
    c for c in df.columns
    if c not in [target, id_col]
]

X = df[feature_cols].copy()
y = df[target].copy()

# =====================================================
# remove nan
# =====================================================

mask = ~X.isna().any(axis=1) & ~y.isna()

X = X[mask]
y = y[mask]

# =====================================================
# standardize
# =====================================================

scaler = StandardScaler()

X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns
)

# =====================================================
# 1. correlation filter
# =====================================================

corr = X_scaled.corr().abs()

plt.figure(figsize=(18,15))

ax = sns.heatmap(
    corr,
    cmap="coolwarm",
    square=True,
    cbar_kws={
        "shrink": 0.8   # 颜色柱缩放
    }
)

# =====================================================
# colorbar 字号
# =====================================================
cbar = ax.collections[0].colorbar

cbar.ax.tick_params(
    labelsize=20   # ⭐颜色柱刻度字号
)

# =====================================================
# 坐标轴字体
# =====================================================
plt.xticks(
    fontsize=12,
    rotation=90
)

plt.yticks(
    fontsize=12,
    rotation=0
)

plt.tight_layout()

plt.savefig(
    "corr_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =====================================================
# correlation filter
# =====================================================
upper = corr.where(
    np.triu(
        np.ones(corr.shape),
        k=1
    ).astype(bool)
)

drop_cols = [
    col for col in upper.columns
    if any(upper[col] > 0.95)
]

X_corr = X_scaled.drop(
    columns=drop_cols
)

print("\n========================")
print("After correlation filter:")
print(X_corr.shape[1], "features")
print("========================")

# =====================================================
# 2. VIF filter
# =====================================================

def calc_vif(df):

    vif = pd.DataFrame()

    vif["Feature"] = df.columns

    vif["VIF"] = [
        variance_inflation_factor(
            df.values,
            i
        )
        for i in range(df.shape[1])
    ]

    return vif

while True:

    vif_df = calc_vif(X_corr)

    max_vif = vif_df["VIF"].max()

    if max_vif < 10:
        break

    remove_feature = vif_df.sort_values(
        "VIF",
        ascending=False
    )["Feature"].iloc[0]

    print("Remove VIF:", remove_feature)

    X_corr = X_corr.drop(
        columns=[remove_feature]
    )

print("\n========================")
print("After VIF filter:")
print(X_corr.shape[1], "features")
print("========================")

# =====================================================
# 3. Boruta
# =====================================================

rf = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)

boruta = BorutaPy(
    estimator=rf,
    n_estimators="auto",
    verbose=2,
    random_state=42
)

boruta.fit(
    X_corr.values,
    y.values
)

selected = X_corr.columns[
    boruta.support_
]

X_boruta = X_corr[selected]

print("\n========================")
print("After Boruta:")
print(len(selected), "features")
print(selected.tolist())
print("========================")

# =====================================================
# 4. RFE
# =====================================================

svr = SVR(kernel="linear")

rfe = RFE(
    estimator=svr,
    n_features_to_select=15
)

rfe.fit(
    X_boruta,
    y
)

final_features = X_boruta.columns[
    rfe.support_
]

X_final = X_boruta[final_features]

print("\n========================")
print("Final selected features:")
print(final_features.tolist())
print("========================")

# =====================================================
# save feature ranking
# =====================================================

rank_df = pd.DataFrame({

    "Feature": X_boruta.columns,

    "RFE_Rank": rfe.ranking_

})

rank_df = rank_df.sort_values(
    "RFE_Rank"
)

rank_df.to_excel(
    "feature_rank.xlsx",
    index=False
)

# =====================================================
# save selected dataset
# =====================================================

final_df = pd.concat(
    [
        X_final,
        y.reset_index(drop=True)
    ],
    axis=1
)

final_df.to_excel(
    "selected_features.xlsx",
    index=False
)

final_df.to_csv(
    "final_selected_features.csv",
    index=False
)

# =====================================================
# Boruta importance
# =====================================================

rf.fit(X_final,y)

imp = pd.DataFrame({

    "Feature": X_final.columns,

    "Importance": rf.feature_importances_

})

imp = imp.sort_values(
    "Importance",
    ascending=False
)

plt.figure(figsize=(10,6))

sns.barplot(
    data=imp,
    x="Importance",
    y="Feature"
)

plt.tight_layout()

plt.savefig(
    "boruta_importance.png",
    dpi=300
)

# =====================================================
# SHAP
# =====================================================

explainer = shap.TreeExplainer(rf)

shap_values = explainer.shap_values(X_final)

# =====================================================
# FEATURE NAMES
# =====================================================

feature_names = X_final.columns.tolist()

# =====================================================
# 1. SHAP SUMMARY (BEESWARM)
# =====================================================

plt.figure(figsize=(10, 7))

shap.summary_plot(
    shap_values,
    X_final,
    show=False
)

plt.tight_layout()

plt.savefig(
    "shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =====================================================
# 2. SHAP BAR IMPORTANCE
# mean(|SHAP|)
# =====================================================

plt.figure(figsize=(10, 7))

shap.summary_plot(
    shap_values,
    X_final,
    plot_type="bar",
    show=False
)

plt.tight_layout()

plt.savefig(
    "shap_bar.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =====================================================
# 3. SHAP DEPENDENCE PLOTS
# Top 5 features
# =====================================================

top5_features = imp["Feature"].head(5).tolist()

for feat in top5_features:

    plt.figure(figsize=(7, 5))

    shap.dependence_plot(
        feat,
        shap_values,
        X_final,
        interaction_index="auto",
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        f"shap_dependence_{feat}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# =====================================================
# 4. SHAP INTERACTION SUMMARY
# =====================================================

print("Computing SHAP interaction values...")

shap_interaction_values = explainer.shap_interaction_values(X_final)

plt.figure(figsize=(10, 7))

shap.summary_plot(
    shap_interaction_values,
    X_final,
    show=False
)

plt.tight_layout()

plt.savefig(
    "shap_interaction_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =====================================================
# 5. SHAP WATERFALL PLOT
# single sample explanation
# =====================================================

sample_id = 10

shap_exp = shap.Explanation(
    values=shap_values[sample_id],
    base_values=explainer.expected_value,
    data=X_final.iloc[sample_id],
    feature_names=feature_names
)

plt.figure(figsize=(8, 6))

shap.plots.waterfall(
    shap_exp,
    show=False
)

plt.tight_layout()

plt.savefig(
    "shap_waterfall_sample10.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =====================================================
# 6. SHAP FORCE PLOT
# =====================================================

force_plot = shap.force_plot(
    explainer.expected_value,
    shap_values[sample_id],
    X_final.iloc[sample_id],
    matplotlib=True,
    show=False
)

plt.savefig(
    "shap_force_sample10.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =====================================================
# 7. SHAP DECISION PLOT
# =====================================================

plt.figure(figsize=(10, 6))

shap.decision_plot(
    explainer.expected_value,
    shap_values[:100],
    X_final.iloc[:100],
    show=False
)

plt.tight_layout()

plt.savefig(
    "shap_decision_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =====================================================
# 8. EXPORT SHAP VALUES
# =====================================================

shap_df = pd.DataFrame(
    shap_values,
    columns=feature_names
)

shap_df.to_csv(
    "shap_values.csv",
    index=False
)

# =====================================================
# 9. MEAN ABS SHAP
# =====================================================

mean_abs_shap = np.abs(shap_values).mean(axis=0)

shap_importance = pd.DataFrame({
    "Feature": feature_names,
    "MeanAbsSHAP": mean_abs_shap
})

shap_importance = shap_importance.sort_values(
    "MeanAbsSHAP",
    ascending=False
)

shap_importance.to_csv(
    "shap_importance.csv",
    index=False
)

# =====================================================
# EXTRA: IMPORTANT FEATURES
# =====================================================

important_features = [

    "Canopy_COV",
    "Canopy_Volume",
    "Canopy_HP",
    "Canopy_LAI",
    "GNDVI_std"

]

for feat in important_features:

    if feat not in X_final.columns:
        continue

    plt.figure(figsize=(7,5))

    shap.dependence_plot(
        feat,
        shap_values,
        X_final,
        interaction_index="auto",
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        f"shap_dependence_{feat}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# =====================================================
# DONE
# =====================================================

print("========================")
print("\n========================")
print("Saved:")
print("- corr_heatmap.png")
print("- boruta_importance.png")
print("- shap_summary.png")
print("- shap_bar.png")
print("- shap_dependence_*.png")
print("- shap_interaction_summary.png")
print("- shap_waterfall_sample0.png")
print("- shap_force_sample0.png")
print("- shap_decision_plot.png")
print("- shap_values.csv")
print("- shap_importance.csv")
print("- selected_features.xlsx")
print("- final_selected_features.csv")
print("- feature_rank.xlsx")
print("========================")