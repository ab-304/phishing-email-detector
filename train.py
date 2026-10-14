"""Train and compare phishing-email classifiers.

Usage:
    python train.py --data emails.csv --text-col "Email Text" --label-col "Email Type"
"""
import argparse
import json
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from features import extract


def to_binary(v):
    s = str(v).strip().lower()
    return 1 if ("phish" in s or "spam" in s or s in ("1", "true")) else 0


def tfidf():
    return TfidfVectorizer(lowercase=True, stop_words="english",
                           ngram_range=(1, 2), min_df=2, max_features=20000,
                           sublinear_tf=True)


def with_features(clf):
    union = FeatureUnion([
        ("tfidf", tfidf()),
        ("hand", Pipeline([("fx", FunctionTransformer(extract)),
                           ("sc", StandardScaler())])),
    ])
    return Pipeline([("features", union), ("clf", clf)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--text-col", default="Email Text")
    ap.add_argument("--label-col", default="Email Type")
    a = ap.parse_args()

    df = pd.read_csv(a.data)[[a.text_col, a.label_col]].dropna()
    df.columns = ["text", "label"]
    df = df.drop_duplicates(subset="text")          # duplicates leak between train/test
    df["y"] = df["label"].map(to_binary)
    print(f"{len(df)} emails | phishing share: {df['y'].mean():.1%}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        df["text"], df["y"], test_size=0.2, random_state=42, stratify=df["y"])

    models = {
        "NaiveBayes (TF-IDF)": Pipeline([("features", tfidf()), ("clf", MultinomialNB())]),
        "LogReg (TF-IDF + features)": with_features(
            LogisticRegression(max_iter=1000, class_weight="balanced")),
        "RandomForest (TF-IDF + features)": with_features(
            RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=42,
                                   class_weight="balanced")),
    }

    rows, fitted = [], {}
    for name, pipe in models.items():
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        p, r, f, _ = precision_recall_fscore_support(y_te, pred, average="binary")
        rows.append({"model": name, "accuracy": accuracy_score(y_te, pred),
                     "phishing_precision": p, "phishing_recall": r, "phishing_f1": f})
        fitted[name] = (pipe, pred)
        print(f"\n=== {name} ===")
        print(classification_report(y_te, pred, target_names=["safe", "phishing"]))

    res = pd.DataFrame(rows).round(4).sort_values("phishing_f1", ascending=False)
    res.to_csv("results.csv", index=False)
    print("\n", res.to_string(index=False))

    best = res.iloc[0]["model"]
    ConfusionMatrixDisplay.from_predictions(
        y_te, fitted[best][1], display_labels=["safe", "phishing"], cmap="Blues")
    plt.title(best)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)

    joblib.dump(fitted[best][0], "model.joblib")
    joblib.dump(fitted["LogReg (TF-IDF + features)"][0], "explain_model.joblib")
    json.dump({"best_model": best}, open("meta.json", "w"))
    print(f"\nSaved best model: {best}")


if __name__ == "__main__":
    main()