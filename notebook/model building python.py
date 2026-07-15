"""
Hybrid Course Recommendation System — training pipeline.

Combines Content-Based Filtering (TF-IDF + cosine similarity) with
Collaborative Filtering (SVD via the `surprise` library), blended with a
weighted alpha formula. Converted from the original Jupyter notebook
(recommendation.ipynb) into a single reusable Python script.

Usage:
    python recommendation.py --data dataset/online_course.xlsx --out Saved_model.pkl
    python recommendation.py --data dataset/online_course.xlsx --sample-user 15796 --top-n 10
"""

import argparse
import logging
import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

TEXT_COLS = ["course_name", "difficulty_level", "instructor",
             "certification_offered", "study_material_available"]
COURSE_COLS = ["course_id", "course_name", "difficulty_level", "instructor",
               "certification_offered", "study_material_available",
               "course_duration_hours", "course_price", "feedback_score"]
NUM_COLS = ["course_duration_hours", "course_price", "feedback_score"]


# --------------------------------------------------------------------------- #
# Data loading & preprocessing
# --------------------------------------------------------------------------- #
def load_data(path: str) -> pd.DataFrame:
    logger.info("Loading dataset from %s", path)
    df = pd.read_excel(path)
    df[TEXT_COLS] = df[TEXT_COLS].fillna("")
    df["rating"] = df["rating"].fillna(df["rating"].median()).clip(1.0, 5.0)
    logger.info("Rows: %d | Courses: %d | Users: %d",
                len(df), df["course_id"].nunique(), df["user_id"].nunique())
    return df


def build_course_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """One row per unique course; numeric features scaled + tokenized; TF-IDF text built."""
    courses = df[COURSE_COLS].drop_duplicates(subset="course_id").reset_index(drop=True)
    courses[TEXT_COLS] = courses[TEXT_COLS].fillna("")

    scaler = MinMaxScaler()
    courses[NUM_COLS] = courses[NUM_COLS].fillna(df[NUM_COLS].median())
    courses[NUM_COLS] = scaler.fit_transform(courses[NUM_COLS])

    # Turn scaled numeric features into text tokens so TF-IDF can compare them
    for col, token in [("course_duration_hours", "duration"),
                        ("course_price", "price"),
                        ("feedback_score", "feedback")]:
        courses[f"{token}_token"] = courses[col].apply(
            lambda x: f"{token}_high" if x > 0.66 else (f"{token}_mid" if x > 0.33 else f"{token}_low")
        )

    # course_name repeated twice to give it more weight in the TF-IDF vocabulary
    courses["features"] = (
        courses["course_name"] + " " + courses["course_name"] + " " +
        courses["difficulty_level"] + " " + courses["instructor"] + " " +
        courses["certification_offered"] + " " + courses["study_material_available"] + " " +
        courses["duration_token"] + " " + courses["price_token"] + " " + courses["feedback_token"]
    )
    logger.info("Built course catalog with %d unique courses", len(courses))
    return courses


def build_tfidf(courses: pd.DataFrame):
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(courses["features"])
    course_index = {cid: idx for idx, cid in enumerate(courses["course_id"])}
    logger.info("TF-IDF matrix: %d courses x %d features", *tfidf_matrix.shape)
    return tfidf, tfidf_matrix, course_index


# --------------------------------------------------------------------------- #
# Collaborative filtering (SVD)
# --------------------------------------------------------------------------- #
def train_cf_model(df: pd.DataFrame):
    cf_df = df[["user_id", "course_id", "rating"]].dropna().copy()
    cf_df["rating"] = cf_df["rating"].clip(1.0, 5.0)

    reader = Reader(rating_scale=(1.0, 5.0))
    data_cf = Dataset.load_from_df(cf_df, reader)
    trainset = data_cf.build_full_trainset()
    logger.info("Trainset: %d users | %d courses | %d ratings",
                trainset.n_users, trainset.n_items, trainset.n_ratings)

    cf_model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
    cf_model.fit(trainset)
    logger.info("SVD model trained.")
    return cf_model, data_cf


def evaluate_cf_model(data_cf, test_size: float = 0.2, random_state: int = 42) -> dict:
    """Held-out 80/20 evaluation (RMSE, MAE). Uses a fresh model so the
    production `cf_model` (trained on the full trainset) is left untouched."""
    train_data, test_data = train_test_split(data_cf, test_size=test_size, random_state=random_state)
    model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=random_state)
    model.fit(train_data)
    predictions = model.test(test_data)

    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    logger.info("CF Model Evaluation (80/20 split) -> RMSE: %.4f | MAE: %.4f", rmse, mae)
    return {"rmse": rmse, "mae": mae}


# --------------------------------------------------------------------------- #
# Hybrid recommendation
# --------------------------------------------------------------------------- #
def normalize_scores(score_dict: dict) -> dict:
    """Min-Max normalize scores to [0, 1] so CB and CF scores sit on the same scale."""
    vals = np.array(list(score_dict.values()))
    mn, mx = vals.min(), vals.max()
    if mx == mn:
        return {k: 0.5 for k in score_dict}
    return {k: (v - mn) / (mx - mn) for k, v in score_dict.items()}


def hybrid_recommend(user_id, df, courses, cf_model, tfidf_matrix, course_index,
                      alpha: float = 0.4, top_n: int = 10, min_rating: float = 3.5,
                      switch_threshold: int = 5) -> pd.DataFrame:
    """
    Generate top-N hybrid recommendations for a user.

    hybrid_score = alpha * CB_norm + (1 - alpha) * CF_norm

    Cold-start handling: users with fewer than `switch_threshold` interactions
    get alpha boosted to 0.7 (more weight on content-based similarity).
    """
    seen = set(df[df["user_id"] == user_id]["course_id"])
    n_seen = len(seen)
    eff_alpha = 0.7 if n_seen < switch_threshold else alpha
    logger.info("User %s | %d interactions | alpha=%.2f", user_id, n_seen, eff_alpha)

    # Content-based scores, seeded from the user's liked courses (rating >= min_rating)
    user_liked = df[(df["user_id"] == user_id) & (df["rating"] >= min_rating)]["course_id"].tolist()
    if not user_liked:
        user_liked = df[df["user_id"] == user_id]["course_id"].tolist()

    cb_raw = np.zeros(len(courses))
    for cid in user_liked:
        if cid in course_index:
            cb_raw += cosine_similarity(tfidf_matrix[course_index[cid]], tfidf_matrix).flatten()
    if user_liked:
        cb_raw /= len(user_liked)
    cb = {cid: cb_raw[course_index[cid]] for cid in courses["course_id"] if cid in course_index}

    # Collaborative filtering scores
    cf = {cid: cf_model.predict(user_id, cid).est for cid in courses["course_id"]}

    # Normalize & blend
    cb_n = normalize_scores(cb)
    cf_n = normalize_scores(cf)
    hybrid = {cid: eff_alpha * cb_n.get(cid, 0) + (1 - eff_alpha) * cf_n.get(cid, 0)
              for cid in courses["course_id"]}

    unseen = {k: v for k, v in hybrid.items() if k not in seen}
    top = sorted(unseen.items(), key=lambda x: x[1], reverse=True)[:top_n]

    rows = []
    for cid, hs in top:
        r = courses[courses["course_id"] == cid].iloc[0]
        rows.append({
            "course_id": cid,
            "course_name": r["course_name"],
            "difficulty_level": r["difficulty_level"],
            "cb_score": round(cb_n.get(cid, 0), 4),
            "cf_score": round(cf_n.get(cid, 0), 4),
            "hybrid_score": round(hs, 4),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_model(cf_model, tfidf_matrix, course_index, courses, alpha, switch_threshold,
               min_rating, out_path: str) -> None:
    artifacts = {
        "cf_model": cf_model,
        "tfidf_matrix": tfidf_matrix,
        "course_index": course_index,
        "courses": courses,
        "alpha": alpha,
        "switch_threshold": switch_threshold,
        "min_rating": min_rating,
    }
    with open(out_path, "wb") as f:
        pickle.dump(artifacts, f)
    logger.info("Model saved to %s", out_path)


# --------------------------------------------------------------------------- #
# CLI entrypoint
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Train and save the hybrid course recommender.")
    parser.add_argument("--data", default="dataset/online_course.xlsx",
                         help="Path to the course dataset (.xlsx)")
    parser.add_argument("--out", default="Saved_model.pkl",
                         help="Output path for the saved model pickle")
    parser.add_argument("--alpha", type=float, default=0.4,
                         help="Content-based weight in the hybrid blend")
    parser.add_argument("--switch-threshold", type=int, default=5,
                         help="Interaction count below which alpha is boosted to 0.7")
    parser.add_argument("--min-rating", type=float, default=3.5,
                         help="Rating threshold above which a course counts as 'liked'")
    parser.add_argument("--sample-user", type=int, default=None,
                         help="Optional user_id to print sample recommendations for")
    parser.add_argument("--top-n", type=int, default=10,
                         help="Number of recommendations to show for --sample-user")
    parser.add_argument("--skip-eval", action="store_true",
                         help="Skip the RMSE/MAE train/test evaluation step")
    args = parser.parse_args()

    df = load_data(args.data)
    courses = build_course_catalog(df)
    tfidf, tfidf_matrix, course_index = build_tfidf(courses)
    cf_model, data_cf = train_cf_model(df)

    if not args.skip_eval:
        evaluate_cf_model(data_cf)

    save_model(cf_model, tfidf_matrix, course_index, courses,
               args.alpha, args.switch_threshold, args.min_rating, args.out)

    if args.sample_user is not None:
        if args.sample_user not in df["user_id"].unique():
            logger.warning("User %s not found in dataset.", args.sample_user)
        else:
            recs = hybrid_recommend(args.sample_user, df, courses, cf_model, tfidf_matrix,
                                     course_index, alpha=args.alpha, top_n=args.top_n,
                                     min_rating=args.min_rating,
                                     switch_threshold=args.switch_threshold)
            print(f"\nTop {args.top_n} Recommendations for User {args.sample_user}\n")
            print(recs.to_string(index=False))


if __name__ == "__main__":
    main()
