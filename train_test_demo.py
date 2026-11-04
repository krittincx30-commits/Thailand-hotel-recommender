"""Hands-on train/test demo (80/20) for K-Means + Decision tree.   Run: python train_test_demo.py
Needs only: pandas numpy scikit-learn  and  thailand_hotels_clustered.csv in the same folder.
Golden rule: FIT on train only, then just APPLY (transform / predict) to test."""
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, accuracy_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, export_text

# ---------------- settings you can play with ----------------
TEST_SIZE = 0.20        # try 0.30
K = 5                   # try 3, 4, 6
SEEDS = range(5)        # more seeds = more reliable average
# ------------------------------------------------------------
df = pd.read_csv("thailand_hotels_clustered.csv")
df["log_price"] = np.log(df.price); df["log_reviews"] = np.log1p(df.n_reviews)
CLUSTER_FEATS = ["log_price", "rating", "room_score", "log_reviews", "pool_villa", "villa", "suite", "bedrooms"]
TREE_FEATS = ["price", "rating", "room_score", "n_reviews", "pool_villa", "villa", "suite", "bedrooms"]   # raw units -> readable rules

rows = []
for seed in SEEDS:
    # 1) split 80/20, keeping the destination mix the same in both parts
    train, test = train_test_split(df, test_size=TEST_SIZE, random_state=seed, stratify=df.destination)

    # 2) K-Means: fit scaler + model on TRAIN only
    scaler = StandardScaler().fit(train[CLUSTER_FEATS])
    Xtr, Xte = scaler.transform(train[CLUSTER_FEATS]), scaler.transform(test[CLUSTER_FEATS])
    km = KMeans(K, n_init=30, random_state=seed).fit(Xtr)
    ytr, yte = km.labels_, km.predict(Xte)                     # test is only PREDICTED, never fitted

    # 3) cluster quality on train vs test (similar numbers = clusters generalise)
    # 4) Decision tree explaining the clusters: pick depth by 5-fold CV on TRAIN, then score on TEST once
    gs = GridSearchCV(DecisionTreeClassifier(min_samples_leaf=10, random_state=seed),
                      {"max_depth": [2, 3, 4, 5]}, cv=5, scoring="f1_macro").fit(train[TREE_FEATS], ytr)
    pred = gs.predict(test[TREE_FEATS])
    rows.append(dict(seed=seed, sil_train=silhouette_score(Xtr, ytr), sil_test=silhouette_score(Xte, yte),
                     db_test=davies_bouldin_score(Xte, yte), ch_test=calinski_harabasz_score(Xte, yte),
                     best_depth=gs.best_params_["max_depth"], cv_f1_train=gs.best_score_,
                     tree_acc_test=accuracy_score(yte, pred), tree_f1_test=f1_score(yte, pred, average="macro")))
    last = (gs, test, yte, pred)

res = pd.DataFrame(rows).round(3)
print(res.to_string(index=False))
print("\nmean +- sd over", len(res), "splits")
for c in res.columns[1:]:
    if c != "best_depth": print(f"  {c:14s} {res[c].mean():.3f} +- {res[c].std():.3f}")

gs, test, yte, pred = last
print("\nconfusion matrix (last split; rows = true cluster, cols = predicted):\n", confusion_matrix(yte, pred))
print("\nrules of the chosen tree (last split):\n", export_text(gs.best_estimator_, feature_names=TREE_FEATS))
