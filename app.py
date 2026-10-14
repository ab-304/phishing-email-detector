import joblib
import numpy as np
import streamlit as st

st.set_page_config(page_title="Phishing Email Detector", page_icon="🎣")
st.title("🎣 Phishing Email Detector")
st.caption("Educational project: paste an email and see the verdict and why.")


@st.cache_resource
def load():
    return joblib.load("model.joblib"), joblib.load("explain_model.joblib")


model, explainer = load()
text = st.text_area("Email content", height=250)

if st.button("Analyze") and text.strip():
    proba = model.predict_proba([text])[0][1]
    if proba >= 0.5:
        st.error(f"Likely PHISHING ({proba:.0%} confidence)")
    else:
        st.success(f"Looks SAFE ({1 - proba:.0%} confidence)")

    # Explanation: which words pushed the logistic regression toward "phishing"
    feats = explainer.named_steps["features"]
    names = feats.transformer_list[0][1].get_feature_names_out()
    coef = explainer.named_steps["clf"].coef_[0][:len(names)]
    vec = feats.transform([text])
    contrib = np.asarray(vec[:, :len(names)].todense()).ravel() * coef
    top = np.argsort(contrib)[::-1][:8]
    top = [i for i in top if contrib[i] > 0]
    if top:
        st.subheader("Words that raised the phishing score")
        st.table({"word": [names[i] for i in top],
                  "weight": [round(float(contrib[i]), 3) for i in top]})