"""Generates cluster-quality metrics, ranking metrics (proxy relevance) and slide figures."""
import random, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score
import recommender as R

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
GREEN, TEAL, GREY = "#0b7236", "#6fa9a5", "#9a9a9a"
d = R.load(); M = R.item_matrix(d)
d["log_price"] = np.log(d.price); d["log_reviews"] = np.log1p(d.n_reviews)
F = ["log_price", "rating", "room_score", "log_reviews", "pool_villa", "villa", "suite", "bedrooms"]
X = StandardScaler().fit_transform(d[F])

# ---------- 1. cluster-quality metrics k=3..7 ----------
rows = []
for k in range(2, 9):
    km = KMeans(k, n_init=30, random_state=42).fit(X)
    rows.append(dict(k=k, inertia=km.inertia_, silhouette=silhouette_score(X, km.labels_),
                     davies_bouldin=davies_bouldin_score(X, km.labels_), calinski_harabasz=calinski_harabasz_score(X, km.labels_)))
cm = pd.DataFrame(rows).round(3); cm.to_csv("cluster_metrics.csv", index=False); print(cm.to_string(index=False))
fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
for a, (c, t, good) in zip(ax, [("inertia", "Elbow (inertia)", ""), ("silhouette", "Silhouette (higher = better)", "max"),
                                ("davies_bouldin", "Davies-Bouldin (lower = better)", "min"), ("calinski_harabasz", "Calinski-Harabasz (higher = better)", "max")]):
    a.plot(cm.k, cm[c], marker="o", color=GREEN); a.set_title(t, fontsize=11); a.set_xlabel("k")
    a.axvline(5, color=TEAL, ls="--", lw=1.5); a.set_xticks(cm.k)
fig.suptitle("Cluster quality metrics by k (dashed line = chosen k = 5)", y=1.04, fontsize=13)
fig.tight_layout(); fig.savefig("figures/fig1_cluster_metrics.png", dpi=200, bbox_inches="tight"); plt.close()

# ---------- 2. cluster profiles ----------
order = d.groupby("cluster_name").price.median().sort_values().index.tolist()
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].boxplot([d[d.cluster_name == c].price for c in order], labels=[c.replace(" ", "\n", 1) for c in order], showfliers=False,
              patch_artist=True, boxprops=dict(facecolor=TEAL), medianprops=dict(color="k"))
ax[0].set_yscale("log"); ax[0].set_ylabel("Price per night (THB, log scale)"); ax[0].set_title("Price by cluster")
for c, col in zip(order, plt.cm.Greens(np.linspace(0.35, 0.95, len(order)))):
    s = d[d.cluster_name == c]; ax[1].scatter(s.price, s.rating, s=14, alpha=0.6, color=col, label=f"{c} (n={len(s)})")
ax[1].set_xscale("log"); ax[1].set_xlabel("Price per night (THB, log)"); ax[1].set_ylabel("Rating"); ax[1].set_title("Rating vs price")
ax[1].legend(fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig("figures/fig2_cluster_profiles.png", dpi=200, bbox_inches="tight"); plt.close()

# ---------- 3. decision tree (depth 3) ----------
TF = ["price", "rating", "room_score", "n_reviews", "pool_villa", "villa", "suite", "bedrooms"]
t = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=42).fit(d[TF], d.cluster)
cv = cross_val_score(DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=42), d[TF], d.cluster, cv=5).mean()
lab = d.groupby("cluster").cluster_name.first().to_dict()
fig, ax = plt.subplots(figsize=(24, 9))
plot_tree(t, feature_names=TF, class_names=[lab[i] for i in sorted(lab)], filled=True, rounded=True, fontsize=8, ax=ax, impurity=False)
ax.set_title(f"Decision tree explaining the 5 clusters (train acc {t.score(d[TF], d.cluster):.2f}, 5-fold CV acc {cv:.2f})", fontsize=13)
fig.savefig("figures/fig3_decision_tree.png", dpi=200, bbox_inches="tight"); plt.close()
print("tree cv", round(cv, 3))

# ---------- 4. ranking metrics with proxy relevance ----------
rng = random.Random(7)
dests = [x for x in d.destination.unique() if (d.destination == x).sum() >= 30]
queries = []
for _ in range(120):
    dest = rng.choice(dests); sub = d[d.destination == dest]
    budget = float(np.quantile(sub.price, rng.choice([0.3, 0.5, 0.8])))
    trav = rng.choice(["solo", "couple", "family", "group"]); style = rng.choice([None, "standard", "suite", "villa"])
    bed = rng.choice([None, "king", "queen", "twin"])
    queries.append(dict(destination=dest, budget=budget, traveler=trav, style=style, bed=bed))

def pool(q):
    m = (d.destination == q["destination"]) & (d.price <= q["budget"])
    if q["traveler"] == "family": m &= d.family_friendly == 1
    return d[m]
def sys_price(q): return pool(q).sort_values("price").head(5)
def sys_comp(q): return pool(q).sort_values("composite_score", ascending=False).head(5)
def sys_cbf(q): return R.recommend(d, q["destination"], q["budget"], q["traveler"], q["style"], q["bed"], "balanced", M=M)

def gain(r, q):  # graded relevance = number of stated preferences met (style, bed)
    g = np.zeros(len(r))
    if q["style"]: g += (r.room_style.values == q["style"])
    if q["bed"]: g += (r.bed_class.values == q["bed"])
    return g
def dcg(g): return float(np.sum(g / np.log2(np.arange(2, len(g) + 2))))

res = []
for name, f in [("A: price sort", sys_price), ("B: composite only", sys_comp), ("C: CBF + composite", sys_cbf)]:
    P, N, MR, REC = [], [], [], []
    for q in queries:
        if not (q["style"] or q["bed"]): continue
        r = f(q)
        if len(r) == 0: continue
        n_pref = bool(q["style"]) + bool(q["bed"]); p = pool(q)
        full = gain(p, q); n_rel = int((full == n_pref).sum())
        if n_rel == 0: continue                       # no relevant item exists -> query not informative
        g = gain(r, q); rel = (g == n_pref)
        P.append(rel.mean()); REC.append(rel.sum() / min(n_rel, 5))
        ideal = np.sort(full)[::-1][:5]; N.append(dcg(g) / dcg(ideal) if dcg(ideal) > 0 else 0)
        MR.append(1 / (np.argmax(rel) + 1) if rel.any() else 0)
    res.append(dict(system=name, queries=len(P), precision_at_5=np.mean(P), recall_at_5_capped=np.mean(REC), ndcg_at_5=np.mean(N), mrr=np.mean(MR)))
rk = pd.DataFrame(res).round(3); rk.to_csv("ranking_metrics.csv", index=False); print(rk.to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 4.2)); w = 0.26; xs = np.arange(4)
for i, (_, r) in enumerate(rk.iterrows()):
    vals = [r.precision_at_5, r.recall_at_5_capped, r.ndcg_at_5, r.mrr]
    b = ax.bar(xs + (i - 1) * w, vals, w, label=r.system, color=[GREY, TEAL, GREEN][i])
    for rect, v in zip(b, vals): ax.text(rect.get_x() + rect.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
ax.set_xticks(xs); ax.set_xticklabels(["Precision@5", "Recall@5\n(capped)", "NDCG@5", "MRR"]); ax.set_ylim(0, 1.05)
ax.set_title(f"Ranking metrics, proxy relevance = matches requested style/bed ({rk.queries.iloc[0]} queries)", fontsize=11); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig("figures/fig4_ranking_metrics.png", dpi=200); plt.close()

# ---------- 5. quality / price trade-off (A vs B vs C) ----------
tr = []
for name, f in [("A: price sort", sys_price), ("B: composite only", sys_comp), ("C: CBF + composite", sys_cbf)]:
    rr = [f(q) for q in queries]; rr = [r for r in rr if len(r)]
    tr.append(dict(system=name, mean_rating=np.mean([r.rating.mean() for r in rr]), mean_price=np.mean([r.price.mean() for r in rr]),
                   diversity=np.mean([r.cluster_name.nunique() / len(r) for r in rr])))
tr = pd.DataFrame(tr).round(3); tr.to_csv("quality_price_tradeoff.csv", index=False); print(tr.to_string(index=False))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for a, col, t_, fmt in [(ax[0], "mean_rating", "Mean rating of top-5", "{:.2f}"), (ax[1], "mean_price", "Mean price of top-5 (THB/night)", "{:,.0f}")]:
    b = a.bar(tr.system.str.replace(": ", ":\n", 1), tr[col], color=[GREY, TEAL, GREEN])
    for rect, v in zip(b, tr[col]): a.text(rect.get_x() + rect.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=9)
    a.set_title(t_, fontsize=11)
ax[0].set_ylim(7, 9.2)
fig.tight_layout(); fig.savefig("figures/fig5_quality_vs_price.png", dpi=200); plt.close()
