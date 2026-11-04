"""Streamlit front-end for the step-by-step hotel recommender.
Run (from this folder):  python -m streamlit run app.py
"""
import os
import streamlit as st
import recommender as R

st.set_page_config(page_title="Thailand Hotel Recommender", page_icon="🏨", layout="wide")


@st.cache_data
def get_data(version):
    d = R.load()
    return d, R.item_matrix(d)


# the version key changes whenever recommender.py or the CSV changes, so old cached data is never reused
_ver = (os.path.getmtime(R.__file__), os.path.getmtime(R.CSV))
d, M = get_data(_ver)
import inspect
if "region" not in d.columns or "strict" not in inspect.signature(R.recommend).parameters:
    st.error("ไฟล์ recommender.py ยังเป็นเวอร์ชันเก่า (หรือยังไม่ได้โหลดใหม่) กรุณาแทนที่ด้วยไฟล์ใหม่ล่าสุด กด Ctrl+C ใน PowerShell แล้วรัน streamlit ใหม่")
    st.stop()
st.title("🏨 ระบบแนะนำโรงแรมในประเทศไทย")
st.caption("ตอบคำถามด้านล่างทีละข้อ แล้วกดปุ่ม \"แนะนำโรงแรม\"")

dests = sorted(d.destination.unique())
dest = st.selectbox("1) จะไปเที่ยวพื้นที่ไหน?", dests, format_func=lambda x: f"{x} ({(d.destination == x).sum()} โรงแรม)")
region = area = None
if dest == "Other Thailand":
    o = d[d.destination == dest]
    cnt = o.region.value_counts()
    regions = [r for r in R.REGION_ORDER if cnt.get(r, 0) > 0]
    region = st.selectbox("ภาค", [None] + regions, format_func=lambda x: f"ทุกภาค ({len(o)} โรงแรม)" if x is None else f"{x} ({cnt[x]} โรงแรม)")
    if region:
        areas = R.other_areas(d, region)
        area = st.selectbox("จังหวัด/อำเภอ", [None] + list(areas.index),
                            format_func=lambda x: f"ทุกพื้นที่ใน{region}" if x is None else f"{x} ({areas[x]} โรงแรม)")
        if region == R.UNKNOWN_REGION:
            st.caption("กลุ่มนี้เป็นหมู่บ้าน/ตำบลที่ระบุภาคไม่ได้จากชื่อ")
sub = d[d.destination == dest]
if region: sub = sub[sub.region == region]
if area: sub = sub[sub.area == area]
cap = 30000
lo, hi = int(sub.price.min()), int(min(sub.price.max(), cap))
if hi <= lo:
    hi = lo + 100
default = int(min(max(sub.price.median(), lo), hi))

style = st.selectbox("2) ประเภทห้อง (จัดกลุ่มจากคำในชื่อห้อง ไม่ใช่ประเภทโรงแรมจริง)", [None] + R.STYLES,
                     format_func=lambda x: "ไม่ระบุ" if x is None else R.STYLE_TH[x])
if style and style != "standard":
    allm = d[d.room_style == style]
    by_dest = allm.destination.value_counts()
    where = " · ".join(f"{k} {v}" for k, v in by_dest.items())
    st.info(f"**{R.STYLE_TH[style]}** มีในข้อมูลทั้งหมด **{len(allm)} แห่ง** กระจายอยู่ที่: {where}")
    if len(allm) <= 15:
        with st.expander(f"ดูรายชื่อ {R.STYLE_TH[style]} ทั้ง {len(allm)} แห่ง", expanded=True):
            st.dataframe(allm.sort_values("price")[["hotel_name", "destination", "area", "price", "rating", "n_reviews"]]
                         .rename(columns={"hotel_name": "โรงแรม", "destination": "พื้นที่", "area": "ย่าน", "price": "ราคา/คืน (บาท)",
                                          "rating": "rating", "n_reviews": "จำนวนรีวิว"}),
                         use_container_width=True, hide_index=True)
        here = int((sub.room_style == style).sum())
        if here == 0:
            st.warning(f"ในพื้นที่ที่เลือกตอนนี้ไม่มี {R.STYLE_TH[style]} เลย ลองเลือกพื้นที่จากรายการด้านบน")

with st.form("answers"):
    c1, c2 = st.columns(2)
    budget = c1.slider("3) งบสูงสุดต่อคืน (บาท)", min_value=lo, max_value=hi, value=default, step=50)
    traveler = c2.radio("4) เดินทางแบบไหน?", ["ไม่ระบุ", "คนเดียว", "คู่รัก", "ครอบครัว", "กลุ่มเพื่อน"], horizontal=True)
    c3, c4 = st.columns(2)
    bed = c3.selectbox("5) ประเภทเตียง", [None] + R.BEDS, format_func=lambda x: "ไม่ระบุ" if x is None else R.BED_TH[x])
    pr_label = c4.radio("6) ให้ความสำคัญกับ", ["สมดุล", "คุณภาพ", "ความคุ้มค่า"], horizontal=True)
    k = st.slider("จำนวนอันดับที่แสดง", 3, 10, 5)
    strict = st.checkbox("แสดงเฉพาะโรงแรมที่ตรงประเภทห้อง/เตียงที่เลือกเท่านั้น", value=True,
                         help="ถ้าไม่ติ๊ก ระบบจะแสดงโรงแรมที่ใกล้เคียงด้วย โดยติดป้ายว่าไม่ตรงเงื่อนไข")
    go = st.form_submit_button("🔎 แนะนำโรงแรม", type="primary")

if go:
    tmap = {"คนเดียว": "solo", "คู่รัก": "couple", "ครอบครัว": "family", "กลุ่มเพื่อน": "group"}
    pr = {"สมดุล": "balanced", "คุณภาพ": "quality", "ความคุ้มค่า": "value"}[pr_label]
    res = R.recommend(d, dest, budget, tmap.get(traveler), style, bed, pr, k=k, M=M, region=region, area=area, strict=strict)
    st.divider()
    if res.empty:
        st.warning("ไม่พบโรงแรมที่ตรงทุกเงื่อนไข")
        if style or bed:
            n, pmin = R.availability(d, dest, style, bed, region, area)
            what = " + ".join(x for x in [R.STYLE_TH[style] if style else None, R.BED_TH[bed] if bed else None] if x)
            if n == 0:
                st.info(f"ในพื้นที่นี้ไม่มีโรงแรมที่เป็น \"{what}\" ในข้อมูลเลย ลองเลือกพื้นที่อื่น หรือเปลี่ยนเงื่อนไขเป็น \"ไม่ระบุ\"")
            else:
                st.info(f"ในพื้นที่นี้มี \"{what}\" {n} แห่ง แต่ราคาเริ่มต้น ~{pmin:,} บาท/คืน (สูงกว่างบที่ตั้งไว้) ลองเพิ่มงบ")
        st.caption("หรือติ๊กออกจาก \"แสดงเฉพาะที่ตรงเงื่อนไข\" เพื่อดูโรงแรมใกล้เคียง")
    else:
        n_pool = int((sub.price <= budget).sum())
        st.subheader(f"แนะนำ {len(res)} อันดับแรก (จากโรงแรม {n_pool} แห่งที่อยู่ในงบ)")
        for i, (_, r) in enumerate(res.iterrows(), 1):
            with st.container(border=True):
                a, b = st.columns([3, 1])
                a.markdown(f"### {i}. {r.hotel_name}")
                a.write(r.why)
                if not (r.style_match and r.bed_match):
                    miss = [t for t, ok in (("ประเภทห้อง", r.style_match), ("ประเภทเตียง", r.bed_match)) if not ok]
                    a.caption("ℹ️ ไม่ตรง" + " และ".join(miss) + "ที่เลือก (แสดงเป็นตัวเลือกใกล้เคียง)")
                if isinstance(r.price_flag, str):
                    a.caption("⚠️ ราคาของโรงแรมนี้ถูกระบุว่าน่าสงสัย ควรตรวจสอบก่อนจอง")
                ws, wq, wv = R.PRIORITY_W[pr]
                b.metric("คะแนนแนะนำรวม", f"{r.final_score:.1f}")
                b.caption(f"= {ws:.2f}×{r.similarity * 100:.0f} + {wq:.2f}×{r.quality_pct:.1f} + {wv:.2f}×{r.value_score:.1f}")
                b.caption(f"ความตรงความต้องการ {r.similarity:.2f}")
        with st.expander("ตารางเปรียบเทียบ"):
            st.dataframe(res[["hotel_name", "price", "rating", "n_reviews", "room_style", "bed_class", "cluster_name",
                              "similarity", "quality_pct", "value_score", "final_score"]].round(2), use_container_width=True)
    with st.expander("📐 คะแนนแต่ละตัวคำนวณอย่างไร"):
        ws, wq, wv = R.PRIORITY_W[pr]
        st.markdown(f"""
**คะแนนแนะนำรวม = {ws:.2f} × ความตรงความต้องการ + {wq:.2f} × คุณภาพ + {wv:.2f} × ความคุ้มค่า**  
(น้ำหนักนี้คือโหมด "{pr_label}" ที่เลือก ทุกตัวเป็นค่า 0–100 ค่าน้ำหนักเป็นค่าที่ผู้พัฒนาตั้งเอง ไม่ได้มาจากข้อมูล)

| ตัวเลข | มาจากไหน |
|---|---|
| **ความตรงความต้องการ** (similarity, 0–1 แล้วคูณ 100) | cosine similarity ระหว่างสิ่งที่เลือก (ประเภทห้อง, เตียง, ผู้เดินทาง) กับคุณสมบัติของโรงแรม 1 = ตรงทุกข้อ, 0 = ไม่ตรงเลย, ถ้าไม่เลือกอะไรเลยจะได้ 1 ทุกแห่ง |
| **quality_pct** (คุณภาพ) | เฉลี่ย rating กับ room_score แบบ Bayesian (ดึงโรงแรมรีวิวน้อยเข้าหาค่าเฉลี่ย) แล้วแปลงเป็นเปอร์เซ็นไทล์เทียบโรงแรมทั้ง 977 แห่ง |
| **value_score** (ความคุ้มค่า) | ยิ่งถูกกว่าโรงแรมในกลุ่ม K-Means เดียวกันยิ่งสูง (เปอร์เซ็นไทล์ราคากลับด้าน) ราคาที่น่าสงสัยได้ 50 |

**ข้อจำกัด:** ประเภทห้องจัดกลุ่มจากชื่อห้อง ไม่ใช่ประเภทโรงแรมจริง · ราคาเป็นราคาประมาณ · ข้อมูล 977 แห่ง แต่ละแห่งมีห้องตัวอย่างห้องเดียว
""")
else:
    st.info("เลือกคำตอบด้านบน แล้วกดปุ่ม \"แนะนำโรงแรม\" ผลลัพธ์จะแสดงตรงนี้")
