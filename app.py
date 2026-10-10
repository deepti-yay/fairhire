import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

st.set_page_config(page_title="FairHire", page_icon="⚖️", layout="wide")

LABELS = {
    "cgpa": "CGPA",
    "projects": "Number of projects",
    "internships": "Internships",
    "certifications": "Certifications",
    "skill_match": "Skill match with job",
    "communication": "Communication score",
    "backlogs": "Backlogs",
    "college_tier": "College tier",
    "career_gap_months": "Career gap",
}


# ---------------------------------------------------------------- data
def generate_data(path="data/hiring_data.csv", n=1500):
    rng = np.random.default_rng(42)
    gender = rng.choice(["Male", "Female"], size=n, p=[0.6, 0.4])
    is_f = (gender == "Female").astype(int)
    cgpa = np.clip(rng.normal(7.6, 1.0, n), 5, 10).round(2)
    projects = np.clip(rng.poisson(2.2, n), 0, 6)
    internships = np.clip(rng.poisson(0.9, n), 0, 3)
    certs = np.clip(rng.poisson(1.5, n), 0, 6)
    skill = np.clip(rng.normal(60, 18, n), 0, 100).round(1)
    comm = np.clip(rng.normal(6.5, 1.5, n), 1, 10).round(1)
    backlogs = np.clip(rng.poisson(0.5, n), 0, 5)
    tier = rng.choice([1, 2, 3], size=n, p=[0.2, 0.4, 0.4])
    gap = np.clip(rng.normal(2 + 5 * is_f, 3, n), 0, 24).round(0)
    merit = (0.9 * (cgpa - 7.5) + 0.35 * projects + 0.6 * internships + 0.15 * certs
             + 0.035 * (skill - 60) + 0.2 * (comm - 6.5) - 0.5 * backlogs - 0.5 * (tier - 1))
    biased = merit - 0.9 * is_f - 0.08 * gap
    prob = 1 / (1 + np.exp(-(biased - 0.3)))
    hired = (rng.random(n) < prob).astype(int)
    df = pd.DataFrame({"gender": gender, "cgpa": cgpa, "projects": projects,
                       "internships": internships, "certifications": certs,
                       "skill_match": skill, "communication": comm, "backlogs": backlogs,
                       "college_tier": tier, "career_gap_months": gap, "hired": hired})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return df


@st.cache_resource(show_spinner="Training models (first load only)...")
def build():
    path = "data/hiring_data.csv"
    df = pd.read_csv(path) if os.path.exists(path) else generate_data(path)
    X = df.drop(columns=["hired", "gender"])
    y = df["hired"]
    g = df["gender"]
    X_tr, X_te, y_tr, y_te, g_tr, g_te = train_test_split(
        X, y, g, test_size=0.2, random_state=42, stratify=y)

    rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42).fit(X_tr, y_tr)
    explainer = shap.TreeExplainer(rf)

    fair = ExponentiatedGradient(
        RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        constraints=DemographicParity())
    fair.fit(X_tr, y_tr, sensitive_features=g_tr)

    def report(name, pred):
        sr = pd.Series(pred, index=g_te.index).groupby(g_te).mean()
        return {"Model": name,
                "Accuracy": round(accuracy_score(y_te, pred), 3),
                "F1": round(f1_score(y_te, pred), 3),
                "Female selection rate": round(sr["Female"], 3),
                "Male selection rate": round(sr["Male"], 3),
                "Demographic parity diff": round(demographic_parity_difference(y_te, pred, sensitive_features=g_te), 3),
                "Equalized odds diff": round(equalized_odds_difference(y_te, pred, sensitive_features=g_te), 3)}

    audit = pd.DataFrame([report("Before (standard model)", rf.predict(X_te)),
                          report("After (bias-mitigated)", fair.predict(X_te))])

    sv = explainer(X_te)
    vals = sv.values[:, :, 1] if sv.values.ndim == 3 else sv.values
    return {"df": df, "X": X, "X_te": X_te, "rf": rf, "fair": fair,
            "explainer": explainer, "audit": audit, "vals": vals}


def shap_for(app, row):
    sv = app["explainer"](row)
    vals = sv.values[0, :, 1] if sv.values.ndim == 3 else sv.values[0]
    base = sv.base_values[0, 1] if np.ndim(sv.base_values) == 2 else sv.base_values[0]
    return vals, base


def reasons(vals, row, cols):
    order = np.argsort(vals)
    pos = [i for i in order[::-1] if vals[i] > 0.01][:3]
    neg = [i for i in order if vals[i] < -0.01][:3]
    fmt = lambda i: f"{LABELS[cols[i]]} = {row.iloc[0, i]:g}"
    return [fmt(i) for i in pos], [fmt(i) for i in neg]



def contrib_chart(vals, row, cols):
    """Horizontal bar chart of each feature's push toward / away from 'hired'."""
    order = np.argsort(np.abs(vals))
    names = [f"{LABELS[cols[i]]} = {row.iloc[0, i]:g}" for i in order]
    v = vals[order]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(names, v, color=["#2E9E6B" if x > 0 else "#D9534F" for x in v])
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("Push toward 'hired' (right, green)  /  toward 'rejected' (left, red)")
    fig.tight_layout()
    return fig


def beeswarm_chart(vals, X):
    """Simple SHAP-style dot plot drawn with plain matplotlib."""
    rng = np.random.default_rng(0)
    order = np.argsort(np.abs(vals).mean(0))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sc = None
    for pos, i in enumerate(order):
        col = X.iloc[:, i].values.astype(float)
        rngv = col.max() - col.min()
        norm = (col - col.min()) / (rngv if rngv else 1)
        y = pos + rng.uniform(-0.28, 0.28, len(col))
        sc = ax.scatter(vals[:, i], y, c=norm, cmap="coolwarm", s=10, alpha=0.8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([LABELS[X.columns[i]] for i in order])
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("Impact on decision (right = toward hired)")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Feature value (low to high)")
    fig.tight_layout()
    return fig

app = build()

# ---------------------------------------------------------------- UI
st.title("⚖️ FairHire")
st.caption("An Explainable and Bias-Aware Hiring Decision Support System  |  Prototype")

t1, t2, t3, t4 = st.tabs(["🧑‍💼 Evaluate candidate", "📊 What drives decisions",
                          "⚖️ Fairness audit", "ℹ️ About"])

# ------------------------------------------------ Tab 1
with t1:
    st.subheader("Enter candidate details")
    c1, c2, c3 = st.columns(3)
    with c1:
        cgpa = st.slider("CGPA", 5.0, 10.0, 7.5, 0.1)
        projects = st.slider("Projects", 0, 6, 2)
        internships = st.slider("Internships", 0, 3, 1)
    with c2:
        certs = st.slider("Certifications", 0, 6, 1)
        skill = st.slider("Skill match with job (%)", 0, 100, 60)
        comm = st.slider("Communication (1-10)", 1.0, 10.0, 6.5, 0.1)
    with c3:
        backlogs = st.slider("Backlogs", 0, 5, 0)
        tier = st.selectbox("College tier", [1, 2, 3], index=1)
        gap = st.slider("Career gap (months)", 0, 24, 0)

    row = pd.DataFrame([{"cgpa": cgpa, "projects": projects, "internships": internships,
                         "certifications": certs, "skill_match": skill,
                         "communication": comm, "backlogs": backlogs,
                         "college_tier": tier, "career_gap_months": gap}])

    model_choice = st.radio("Model", ["Standard model", "Bias-mitigated model"], horizontal=True)

    if st.button("Evaluate candidate", type="primary"):
        p = app["rf"].predict_proba(row)[0, 1]
        if model_choice == "Bias-mitigated model":
            pred = int(app["fair"].predict(row)[0])
        else:
            pred = int(p >= 0.5)

        st.divider()
        m1, m2 = st.columns([1, 2])
        with m1:
            if pred == 1:
                st.success("### ✅ Recommended for shortlist")
            else:
                st.error("### ❌ Not recommended")
            st.metric("Model confidence (standard model)", f"{p*100:.0f}%")
            st.caption("Decision support only. A human recruiter should make the final call.")

        vals, base = shap_for(app, row)
        pos, neg = reasons(vals, row, list(row.columns))
        with m2:
            st.markdown("#### Why this result?")
            st.markdown("**Helping the candidate:** " + (", ".join(pos) if pos else "no strong positives"))
            st.markdown("**Hurting the candidate:** " + (", ".join(neg) if neg else "no strong negatives"))
            if gap >= 6:
                st.warning("Career gap is a proxy for gender in this data. See the Fairness audit tab.")

        st.pyplot(contrib_chart(vals, row, list(row.columns)), clear_figure=True)

# ------------------------------------------------ Tab 2
with t2:
    st.subheader("Which features matter most overall?")
    imp = pd.Series(np.abs(app["vals"]).mean(0), index=[LABELS[c] for c in app["X"].columns]).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(imp.index, imp.values, color="#4C78A8")
    ax.set_xlabel("Average impact on decision (SHAP)")
    st.pyplot(fig, clear_figure=True)
    st.write("Career gap appears in the top features. That matters, because it is "
             "linked to gender in this dataset (see the Fairness audit).")
    st.subheader("Detailed view")
    st.pyplot(beeswarm_chart(app["vals"], app["X_te"]), clear_figure=True)

# ------------------------------------------------ Tab 3
with t3:
    st.subheader("Does the model treat men and women equally?")
    st.write("Gender is **not** a model input. We only use it to audit the results.")
    a = app["audit"]
    st.dataframe(a, hide_index=True, width="stretch")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(2)
    ax.bar(x - 0.18, a["Female selection rate"], 0.36, label="Female", color="#E45756")
    ax.bar(x + 0.18, a["Male selection rate"], 0.36, label="Male", color="#4C78A8")
    ax.set_xticks(x)
    ax.set_xticklabels(["Before", "After"])
    ax.set_ylabel("Shortlist rate")
    ax.legend()
    st.pyplot(fig, clear_figure=True)

    d0, d1 = a["Demographic parity diff"][0], a["Demographic parity diff"][1]
    st.info(f"The selection gap between men and women went from **{d0:.3f}** to **{d1:.3f}**. "
            f"Lower is fairer (0 means equal selection rates).")
    st.write("**Why bias existed even without gender as an input:** women in this data have longer "
             "average career gaps, so the model learned gender indirectly through that feature (a proxy).")
    g = app["df"].groupby("gender").career_gap_months.mean().round(1)
    st.write(f"Average career gap: Female **{g['Female']}** months, Male **{g['Male']}** months.")

# ------------------------------------------------ Tab 4
with t4:
    st.subheader("About this prototype")
    st.markdown("""
**Problem (IBM AIML PS 50):** hiring decisions should be transparent and fair.

**What it does**
1. Predicts whether a candidate should be shortlisted
2. Explains every decision with SHAP
3. Audits the model for gender bias with Fairlearn
4. Reduces the bias with a mitigation step

**Tech:** Python, scikit-learn, SHAP, Fairlearn, Streamlit

**Limitations**
- Data is synthetic, with bias planted on purpose to demonstrate the method
- Only one sensitive attribute (gender) and one fairness measure (demographic parity)
- Fairness tools reduce bias but do not remove it fully
- Not for real hiring without human review and real, audited data
""")
