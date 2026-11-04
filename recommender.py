"""Step-by-step content-based hotel recommender (Thailand).
Run:  python recommender.py        (interactive text Q&A)
"""
import re, numpy as np, pandas as pd
from pathlib import Path

CSV = Path(__file__).with_name("thailand_hotels_clustered.csv")
STYLES = ["standard", "suite", "villa", "pool_villa"]
BEDS = ["king", "queen", "double", "twin", "multi"]
CAPS = ["cap2", "cap3_4", "cap5p"]
STYLE_TH = {"standard": "ห้องทั่วไป", "suite": "Suite/Studio/Apartment", "villa": "Villa/Bungalow", "pool_villa": "Pool villa"}
BED_TH = {"king": "เตียงคิง", "queen": "เตียงควีน", "double": "เตียงคู่/ฟูล", "twin": "เตียงแฝด", "multi": "หลายเตียง"}
TRAVELER_CAP = {"solo": "cap2", "couple": "cap2", "family": "cap3_4", "group": "cap5p"}
PRIORITY_W = {"balanced": (0.30, 0.50, 0.20), "quality": (0.20, 0.65, 0.15), "value": (0.20, 0.30, 0.50)}  # (similarity, quality, value)

# ---- region grouping for the "Other Thailand" bucket (hand-mapped from area names; villages not listed -> UNKNOWN_REGION) ----
UNKNOWN_REGION = "ไม่ระบุภาค"
_REGIONS = {
    "ภาคเหนือ": "Chiang Rai;Chiang Khong;Chiang Saen;Mae Sai;Mae Salong;Mae Salong Nai;Mae Hong Son;Ban Khun Yuam;Pang Mapha;Mae Sariang;Phrae;Nan;Lampang;Uttaradit;Phayao;Chiang Kham;Chiang Muan;Phitsanulok;Phetchabun;Sukhothai;Sawankhalok;Mueang Kao;Tak;Mae Sot;Li;Mon Jam",
    "ภาคอีสาน": "Udon Thani;Nong Khai;Loei;Ubon Ratchathani;Nakhon Ratchasima;Buriram;Mukdahan;Yasothon;Roi Et;Si Sa Ket;Nakhon Phanom;That Phanom;Khong Chiam;Maha Sarakham;Phimai;Nang Rong;Chaiyaphum;Dan Sai;Pak Chong;Khon Kaen;Bung Kan;Sakon Nakhon;Kalasin;Nong Bua Lamphu;Amphoe Kumphawapi;Chiang Khan;Amphoe Chiang Khan;Chiang Khan Walking Street;Phu Wiang;Phu Khieo;Phu Ruea;Sung Noen;Ban Phai;Non Sang;Seka;Det Udom;Si Bun Ruang;Phayayen;Prakhon Chai;Khun Han;Trakan Phut Phon",
    "ภาคกลาง": "Nakhon Pathom;Lop Buri;Sara Buri;Uthai Thani;Phra Nakhon Si Ayutthaya;Ang Thong;Sing Buri;Chai Nat;Nakhon Nayok;Bang Bo;Sathorn;Ratchadaphisek;Ta Khli;Muak Lek District;Ban Don Muang;Ban Dan Chang",
    "ภาคตะวันออก": "Chachoengsao;Chanthaburi;Sa Kaeo;Prachin Buri;Aranyaprathet;Kabin Buri;Pluak Daeng;Mae Pim;Baan Khai;Na Kluea;Nong Prue;304 Industrial Park;Ban Khao Hin Son;Watthana Nakhon",
    "ภาคตะวันตก": "Ratchaburi;Kanchanaburi City;Thong Pha Phum;Sai Yok;Sangkhla Buri;Suan Phung;Si Sawat;Hin Dat;Pak Nam Pran",
    "ภาคใต้": "Suratthani;Chumphon;Songkhla;Nakhon Si Thammarat;Khanom;Sichon;Don Sak;Pathiu;Narathiwat;Sungai Kolok;Lang Suan;Dannok;Ko Sukon;Salad Beach;Ban Pak Nam Lamae (1)",
}
REGION_OF = {a: r for r, names in _REGIONS.items() for a in names.split(";")}
REGION_ORDER = list(_REGIONS) + [UNKNOWN_REGION]


def other_areas(d, region, min_hotels=2):
    """Areas inside the 'Other Thailand' bucket for one region (areas with >= min_hotels hotels)."""
    sub = d[(d.destination == "Other Thailand") & (d.region == region)]
    vc = sub.area.value_counts()
    return vc[vc >= min_hotels]


def _capacity(bed: str, room: str, bedrooms: float) -> int:
    b = str(bed).lower()
    if b.startswith("multiple"):
        cap = 4
    else:
        cap = 0
        for part in re.findall(r"(\d+)\s+(king|queen|full|twin|single|sofa|bunk|double)", b):
            n, kind = int(part[0]), part[1]
            cap += n * (2 if kind in ("king", "queen", "full", "double") else 1)
        if "double or 2 twins" in b:
            cap = 2
        if cap == 0:
            cap = 2
    r = str(room).lower()
    if re.search(r"quadruple|quad", r): cap = max(cap, 4)
    if "triple" in r: cap = max(cap, 3)
    if "family" in r: cap = max(cap, 4)
    return int(max(cap, 2 * bedrooms))


def _bed_class(bed: str) -> str:
    b = str(bed).lower()
    if b.startswith("multiple") or re.match(r"[3-9] beds", b) or b.startswith("2 beds"): return "multi"
    if "king" in b: return "king"
    if "queen" in b: return "queen"
    if "full" in b or "double or" in b: return "double"
    if "twin" in b: return "twin"
    return "double"


def load() -> pd.DataFrame:
    d = pd.read_csv(CSV)
    d["room_style"] = np.select([d.pool_villa == 1, d.villa == 1, d.suite == 1], ["pool_villa", "villa", "suite"], "standard")
    d["bed_class"] = d.bed_type.map(_bed_class)
    d["region"] = np.where(d.destination == "Other Thailand", d.area.map(REGION_OF).fillna(UNKNOWN_REGION), None)
    d["capacity"] = [_capacity(b, r, n) for b, r, n in zip(d.bed_type, d.room_type, d.bedrooms)]
    d["cap_class"] = np.select([d.capacity <= 2, d.capacity <= 4], ["cap2", "cap3_4"], "cap5p")
    r = d.room_type.str.lower()
    d["family_friendly"] = ((d.bedrooms >= 2) | (d.capacity >= 4) | r.str.contains("family|triple|quadruple")).astype(int)
    return d


def _onehot(series, cats):
    return np.array([[1.0 if v == c else 0.0 for c in cats] for v in series])


def item_matrix(d):
    return np.hstack([_onehot(d.room_style, STYLES), _onehot(d.bed_class, BEDS), _onehot(d.cap_class, CAPS)])


def user_vector(style=None, bed=None, traveler=None):
    v = np.zeros(len(STYLES) + len(BEDS) + len(CAPS))
    if style: v[STYLES.index(style)] = 1
    if bed: v[len(STYLES) + BEDS.index(bed)] = 1
    if traveler: v[len(STYLES) + len(BEDS) + CAPS.index(TRAVELER_CAP[traveler])] = 1
    return v


def recommend(d, destination, budget, traveler=None, style=None, bed=None, priority="balanced", k=5, area_kw=None, M=None, region=None, area=None, strict=False):
    M = item_matrix(d) if M is None else M
    mask = (d.destination == destination) & (d.price <= budget)
    if region:
        mask &= d.region == region
    if area:
        mask &= d.area == area
    if area_kw:
        mask &= d.location.str.contains(area_kw, case=False, na=False) | d.area.str.contains(area_kw, case=False, na=False)
    if traveler == "family":
        mask &= d.family_friendly == 1
    if strict:  # only keep hotels that exactly match the room type / bed the user asked for
        if style: mask &= d.room_style == style
        if bed: mask &= d.bed_class == bed
    cand = d[mask].copy()
    if cand.empty: return cand
    u = user_vector(style, bed, traveler)
    Mi = M[mask.values]
    if u.any():
        cand["similarity"] = (Mi @ u) / (np.linalg.norm(Mi, axis=1) * np.linalg.norm(u))
    else:
        cand["similarity"] = 1.0
    ws, wq, wv = PRIORITY_W[priority]
    cand["final_score"] = ws * cand.similarity * 100 + wq * cand.quality_pct + wv * cand.value_score
    cand["style_match"] = (cand.room_style == style) if style else True
    cand["bed_match"] = (cand.bed_class == bed) if bed else True
    cand = cand.sort_values(["final_score", "n_reviews"], ascending=False).head(k)
    cand["why"] = [explain(r) for _, r in cand.iterrows()]
    return cand


def availability(d, destination, style=None, bed=None, region=None, area=None):
    """Number of hotels in the chosen place with this room type / bed, ignoring budget (for 'no result' messages)."""
    m = d.destination == destination
    if region: m &= d.region == region
    if area: m &= d.area == area
    if style: m &= d.room_style == style
    if bed: m &= d.bed_class == bed
    sub = d[m]
    return len(sub), (int(sub.price.min()) if len(sub) else None)


def explain(r):
    return (f"{STYLE_TH[r.room_style]}, {BED_TH[r.bed_class]}, ราคา ~{int(r.price):,} บาท/คืน, "
            f"rating {r.rating:.1f} จาก {int(r.n_reviews):,} รีวิว, ความคุ้มค่า {r.value_score:.0f}/100, กลุ่ม: {r.cluster_name}")


def _ask(prompt, options=None, allow_skip=True):
    while True:
        s = input(prompt).strip()
        if not s and allow_skip: return None
        if options is None: return s
        if s.isdigit() and 1 <= int(s) <= len(options): return options[int(s) - 1]
        print("  กรุณาเลือกหมายเลขตามตัวเลือก")


def main():
    d = load(); M = item_matrix(d)
    dests = sorted(d.destination.unique())
    print("== ระบบแนะนำโรงแรมในประเทศไทย ==")
    for i, x in enumerate(dests, 1): print(f" {i}. {x} ({(d.destination == x).sum()} โรงแรม)")
    dest = _ask("1) เลือกพื้นที่ (หมายเลข): ", dests, allow_skip=False)
    kw = _ask("   ระบุย่าน/จังหวัดเพิ่ม (Enter = ข้าม): ") if dest == "Other Thailand" else None
    budget = float(_ask("2) งบสูงสุดต่อคืน (บาท): ", None, allow_skip=False))
    trav = _ask("3) เดินทางแบบไหน? 1.คนเดียว 2.คู่รัก 3.ครอบครัว 4.กลุ่มเพื่อน (Enter = ข้าม): ", ["solo", "couple", "family", "group"])
    style = _ask("4) ประเภทห้อง (จัดกลุ่มจากชื่อห้อง)? 1.ห้องทั่วไป 2.Suite/Studio 3.Villa/Bungalow 4.Pool villa (Enter = ข้าม): ", STYLES)
    bed = _ask("5) เตียง? 1.คิง 2.ควีน 3.ฟูล/คู่ 4.แฝด 5.หลายเตียง (Enter = ข้าม): ", BEDS)
    pr = _ask("6) ให้ความสำคัญ? 1.สมดุล 2.คุณภาพ 3.คุ้มค่า (Enter = สมดุล): ", ["balanced", "quality", "value"]) or "balanced"
    res = recommend(d, dest, budget, trav, style, bed, pr, area_kw=kw, M=M)
    if res.empty:
        print("\nไม่พบโรงแรมตามเงื่อนไข ลองเพิ่มงบหรือเลือกพื้นที่อื่น"); return
    print(f"\nโรงแรมแนะนำ {len(res)} อันดับแรก:")
    for i, (_, r) in enumerate(res.iterrows(), 1):
        print(f" {i}. {r.hotel_name}\n    {r.why}\n    คะแนนแนะนำรวม {r.final_score:.1f} (ความตรงความต้องการ {r.similarity:.2f})")


if __name__ == "__main__":
    main()
