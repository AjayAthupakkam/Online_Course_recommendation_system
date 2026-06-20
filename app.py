import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity



# Hide the GitHub icon, the main menu, and the header toolbar entirely
hide_github_and_menu = """
    <style>
    /* Hides the GitHub icon and the edit/star buttons */
    [data-testid="stHeader"] {{
        visibility: hidden;
        height: 0rem;
    }}
    /* Alternatively, to specifically target just the GitHub icon if it remains */
    button[title="View code on GitHub"] {
        display: none !important;
    }
    </style>
"""
st.markdown(hide_github_and_menu, unsafe_allow_html=True)




# ── PAGE CONFIG ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Course Recommender",
    page_icon  = "🎓",
    layout     = "wide"
)

# ── LOAD ARTIFACTS ────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("model_artifacts.pkl", "rb") as f:
        return pickle.load(f)

artifacts    = load_artifacts()
course_dedup = artifacts["course_dedup"]
user_profile = artifacts["user_profile"]
user_exp     = artifacts["user_exp"]
course_meta  = artifacts["course_meta"]
feature_cols = artifacts["feature_cols"]
user_taken   = artifacts["user_taken"]

# ── DIFFICULTY RULE ───────────────────────────────────────────────────────
def get_difficulty(exp):
    if exp <= 3:
        return ["Beginner"]
    elif exp <= 10:
        return ["Beginner", "Intermediate"]
    else:
        return ["Intermediate", "Advanced"]

# ── RECOMMEND FUNCTION ────────────────────────────────────────────────────
def recommend(user_id, top_n=5):
    if user_id not in user_profile.index:
        return None, "User ID not found in the system."

    profile_vec  = user_profile.loc[user_id].values.reshape(1, -1)
    scores       = cosine_similarity(profile_vec, course_dedup.values)[0]
    score_series = pd.Series(scores, index=course_dedup.index)

    taken        = user_taken.get(user_id, [])
    score_series = score_series.drop(index=taken, errors="ignore")

    exp             = int(user_exp.get(user_id, 3))
    allowed         = get_difficulty(exp)
    allowed_courses = course_meta[course_meta["difficulty_level"].isin(allowed)].index
    score_series    = score_series[score_series.index.isin(allowed_courses)]
    score_series    = score_series[score_series >= 0.50]

    if score_series.empty:
        return None, "No recommendations found for this user."

    top    = score_series.sort_values(ascending=False).head(top_n)
    result = course_meta.loc[top.index].copy()
    result["similarity_score"] = top.values.round(4)
    result = result.reset_index()
    result.columns = ["course_id","course_name","difficulty_level","rating","course_price","similarity_score"]
    return result, None

# ── SIDEBAR ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Course Recommender")
    st.markdown("---")
    st.markdown("**How it works**")
    st.markdown("1. Enter your User ID")
    st.markdown("2. System reads your course history")
    st.markdown("3. Builds your interest profile")
    st.markdown("4. Recommends best matching courses")
    st.markdown("---")
    st.markdown("**Model Type**")
    st.info("Content-Based + Knowledge-Based Hybrid")
    st.markdown("**Similarity Metric**")
    st.info("Cosine Similarity")
    st.markdown("---")
    st.markdown(f"👥 Total Users: **{len(user_profile):,}**")
    st.markdown(f"📚 Total Courses: **{len(course_dedup):,}**")

# ── MAIN ──────────────────────────────────────────────────────────────────
st.title("🎓 Online Course Recommendation System")
st.markdown("Personalized course suggestions based on your learning history and interests.")
st.markdown("---")

# ── INPUT ROW ─────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    user_id_input = st.number_input(
        "Enter User ID",
        min_value = 1,
        max_value = int(user_profile.index.max()),
        value     = 1,
        step      = 1,
        help      = f"Enter any User ID from 1 to {int(user_profile.index.max())}"
    )
with c2:
    top_n = st.selectbox("No. of Recommendations", [5, 10, 15, 20])
with c3:
    st.markdown("<br>", unsafe_allow_html=True)
    btn = st.button("🔍 Recommend", use_container_width=True)

# ── USER SUMMARY ──────────────────────────────────────────────────────────
uid = int(user_id_input)
if uid in user_profile.index:
    exp   = int(user_exp.get(uid, 3))
    level = get_difficulty(exp)
    taken = user_taken.get(uid, [])

    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("👤 User ID",           uid)
    m2.metric("📚 Courses Completed",  exp)
    m3.metric("🎯 Eligible Levels",    " & ".join(level))
    m4.metric("📋 Courses Taken",      len(taken))

    # show taken courses
    with st.expander("📖 View Courses Already Taken"):
        taken_info = course_meta[course_meta.index.isin(taken)][["course_name","difficulty_level","rating"]].reset_index()
        taken_info.columns = ["Course ID","Course Name","Difficulty","Rating"]
        taken_info["Rating"] = taken_info["Rating"].apply(lambda x: f"⭐ {x:.1f}")
        st.dataframe(taken_info, use_container_width=True, hide_index=True)

# ── RECOMMENDATIONS ───────────────────────────────────────────────────────
if btn:
    uid = int(user_id_input)
    with st.spinner("🔄 Finding best courses for you..."):
        results, error = recommend(uid, top_n)

    st.markdown("---")
    if error:
        st.error(f"⚠️ {error}")
    else:
        st.subheader(f"✅ Top {top_n} Recommended Courses for User {uid}")

        # display table
        display = results.copy()

        diff_icon = {"Beginner":"🟢","Intermediate":"🟡","Advanced":"🔴"}
        display["difficulty_level"] = display["difficulty_level"].apply(
            lambda x: f"{diff_icon.get(x,'⚪')} {x}"
        )
        display["rating"]           = display["rating"].apply(lambda x: f"⭐ {x:.1f}")
        display["course_price"]     = display["course_price"].apply(lambda x: f"₹ {x:,.0f}")
        display["similarity_score"] = display["similarity_score"].apply(lambda x: f"{x:.4f}")
        display.insert(0, "Rank", [f"# {i+1}" for i in range(len(display))])
        display = display.drop(columns=["course_id"])
        display.columns = ["Rank","Course Name","Difficulty","Rating","Price","Similarity Score"]

        st.dataframe(display, use_container_width=True, hide_index=True)

        # ── METRICS ROW ───────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📊 Recommendation Summary")
        diff_counts = results["difficulty_level"].value_counts()

        cols = st.columns(len(diff_counts) + 1)
        for i, (diff, count) in enumerate(diff_counts.items()):
            cols[i].metric(f"{diff_icon.get(diff,'⚪')} {diff}", f"{count} courses")

        exp = int(user_exp.get(uid, 3))
        cols[-1].metric("📈 Experience Level", f"{exp} / 19 courses")
        st.progress(min(exp / 19, 1.0))

# ── EXPLORE COURSES ───────────────────────────────────────────────────────
st.markdown("---")
with st.expander("🔎 Explore All Courses in Catalog"):
    diff_filter = st.multiselect(
        "Filter by Difficulty",
        options = ["Beginner","Intermediate","Advanced"],
        default = ["Beginner","Intermediate","Advanced"]
    )
    catalog = course_meta[course_meta["difficulty_level"].isin(diff_filter)].copy().reset_index()
    catalog.columns = ["Course ID","Course Name","Difficulty","Rating","Price (₹)"]
    catalog["Rating"]   = catalog["Rating"].apply(lambda x: f"⭐ {x:.1f}")
    catalog["Price (₹)"] = catalog["Price (₹)"].apply(lambda x: f"₹ {x:,.0f}")
    st.dataframe(catalog, use_container_width=True, hide_index=True)
    st.caption(f"Showing {len(catalog):,} courses")
