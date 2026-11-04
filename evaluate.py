import numpy as np, pandas as pd, random
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
import recommender as R
d = R.load(); M = R.item_matrix(d)

# ---------- RQ2: stability + Family check ----------
F = ['log_price','rating','room_score','log_reviews','pool_villa','villa','suite','bedrooms']
d['log_price']=np.log(d.price); d['log_reviews']=np.log1p(d.n_reviews)
X = StandardScaler().fit_transform(d[F])
base = KMeans(5, n_init=50, random_state=42).fit_predict(X)
ari_seed = [adjusted_rand_score(base, KMeans(5, n_init=10, random_state=s).fit_predict(X)) for s in range(20)]
rng = np.random.default_rng(0); ari_boot = []
for _ in range(30):
    idx = rng.choice(len(d), len(d)); km = KMeans(5, n_init=10, random_state=0).fit(X[idx])
    ari_boot.append(adjusted_rand_score(base, km.predict(X)))
print('ARI seeds mean %.3f min %.3f | bootstrap mean %.3f min %.3f' % (np.mean(ari_seed), min(ari_seed), np.mean(ari_boot), min(ari_boot)))
print(pd.crosstab(d.cluster_name, d.family_friendly, margins=True))
print(d.groupby('cluster_name').family_friendly.mean().round(2))

# ---------- RQ1 / RQ3: 120 simulated queries, 3 systems ----------
rng = random.Random(7)
dests = [x for x in d.destination.unique() if (d.destination == x).sum() >= 30]
queries = []
for _ in range(120):
    dest = rng.choice(dests); sub = d[d.destination == dest]
    budget = float(np.quantile(sub.price, rng.choice([0.3, 0.5, 0.8])))
    trav = rng.choice(['solo','couple','family','group']); style = rng.choice([None,'standard','suite','villa'])
    bed = rng.choice([None,'king','queen','twin'])
    queries.append(dict(destination=dest, budget=budget, traveler=trav, style=style, bed=bed))

def sys_price(q):
    m = (d.destination==q['destination'])&(d.price<=q['budget'])
    if q['traveler']=='family': m &= d.family_friendly==1
    return d[m].sort_values('price').head(5)
def sys_comp(q):
    m = (d.destination==q['destination'])&(d.price<=q['budget'])
    if q['traveler']=='family': m &= d.family_friendly==1
    return d[m].sort_values('composite_score', ascending=False).head(5)
def sys_cbf(q):
    return R.recommend(d, q['destination'], q['budget'], q['traveler'], q['style'], q['bed'], 'balanced', M=M)

rows = []
for name, f in [('A price-sort', sys_price), ('B composite-only', sys_comp), ('C CBF+composite', sys_cbf)]:
    for q in queries:
        r = f(q); n = len(r)
        if n == 0:
            rows.append(dict(system=name, returned5=0, budget_ok=np.nan, style_match=np.nan, bed_match=np.nan, family_ok=np.nan, diversity=np.nan, mean_rating=np.nan, low_review_share=np.nan, mean_price=np.nan, n_reviews_med=np.nan)); continue
        fam = q['traveler'] != 'family'
        rows.append(dict(system=name, returned5=int(n >= 5), budget_ok=(r.price <= q['budget']).mean(),
            style_match=(r.room_style == q['style']).mean() if q['style'] else np.nan,
            bed_match=(r.bed_class == q['bed']).mean() if q['bed'] else np.nan,
            family_ok=r.family_friendly.mean() if q['traveler']=='family' else np.nan,
            diversity=r.cluster_name.nunique()/n, mean_rating=r.rating.mean(),
            low_review_share=(r.n_reviews < 20).mean(), mean_price=r.price.mean(), n_reviews_med=r.n_reviews.median()))
res = pd.DataFrame(rows)
summ = res.groupby('system').mean(numeric_only=True).round(3)
print(summ.T.to_string())
summ.to_csv('evaluation_results.csv')

# ---------- human-evaluation sheet (pairwise A vs C, 12 queries) ----------
sheet = []
for i, q in enumerate(queries[:12], 1):
    a = sys_price(q).hotel_name.tolist(); c = sys_cbf(q).hotel_name.tolist()
    sheet.append(dict(query_id=i, **q, list_A_price_sort=' | '.join(a), list_C_recommender=' | '.join(c), preferred_list_AorC='', comment=''))
pd.DataFrame(sheet).to_csv('human_eval_sheet.csv', index=False, encoding='utf-8-sig')

# ---------- achievable-match (normalised by what the candidate pool allows) ----------
def pool(q):
    m = (d.destination==q['destination'])&(d.price<=q['budget'])
    if q['traveler']=='family': m &= d.family_friendly==1
    return d[m]
out = {}
for name, f in [('A price-sort', sys_price), ('B composite-only', sys_comp), ('C CBF+composite', sys_cbf)]:
    s=[]; b=[]
    for q in queries:
        p = pool(q); r = f(q)
        if len(r)==0: continue
        if q['style']:
            cap = min(5, (p.room_style==q['style']).sum())
            if cap>0: s.append((r.room_style==q['style']).sum()/cap)
        if q['bed']:
            cap = min(5, (p.bed_class==q['bed']).sum())
            if cap>0: b.append((r.bed_class==q['bed']).sum()/cap)
    out[name] = dict(style_vs_ceiling=round(np.mean(s),3), bed_vs_ceiling=round(np.mean(b),3), n_style=len(s), n_bed=len(b))
print(pd.DataFrame(out).T)
pd.DataFrame(out).T.to_csv('evaluation_normalised.csv')
