import re
import itertools
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

st.set_page_config(page_title="FairHire | Explainable Hiring Support", page_icon="⚖️", layout="wide")

# =====================================================================
# Knowledge base: job roles, required skills, resume keywords
# =====================================================================
ROLES = {
    "Data Analyst": ["Python", "SQL", "Excel", "Statistics", "Power BI", "Data Visualization", "Pandas", "Data Cleaning"],
    "Machine Learning Engineer": ["Python", "scikit-learn", "Machine Learning", "Statistics", "SQL", "Git", "Deep Learning", "Pandas"],
    "Web Developer": ["HTML", "CSS", "JavaScript", "React", "Git", "REST APIs", "SQL", "Node.js"],
    "Backend Developer": ["Python", "Java", "SQL", "REST APIs", "Git", "Docker", "Data Structures", "System Design"],
    "Cloud / DevOps Engineer": ["Linux", "Docker", "Kubernetes", "AWS", "Git", "CI/CD", "Python", "Networking"],
}
ALL_SKILLS = sorted({s for v in ROLES.values() for s in v})

ALIASES = {
    "Python": ["python"], "SQL": ["sql", "mysql", "postgresql"], "Excel": ["excel"],
    "Statistics": ["statistics", "statistical"], "Power BI": ["power bi", "powerbi"],
    "Data Visualization": ["data visualization", "data visualisation", "matplotlib", "seaborn", "tableau"],
    "Pandas": ["pandas"], "Data Cleaning": ["data cleaning", "data cleansing", "data wrangling"],
    "scikit-learn": ["scikit-learn", "scikit learn", "sklearn"],
    "Machine Learning": ["machine learning", "ml"],
    "Deep Learning": ["deep learning", "neural network", "neural networks", "tensorflow", "pytorch", "keras"],
    "Git": ["git", "github", "gitlab"], "HTML": ["html", "html5"], "CSS": ["css", "css3"],
    "JavaScript": ["javascript", "js"], "React": ["react", "reactjs", "react.js"],
    "REST APIs": ["rest api", "rest apis", "restful", "rest"], "Node.js": ["node.js", "nodejs", "node js"],
    "Java": ["java"], "Docker": ["docker"], "Data Structures": ["data structures", "dsa", "algorithms"],
    "System Design": ["system design"], "Linux": ["linux", "ubuntu"], "Kubernetes": ["kubernetes", "k8s"],
    "AWS": ["aws", "amazon web services"], "CI/CD": ["ci/cd", "cicd", "jenkins", "github actions"],
    "Networking": ["networking", "tcp/ip", "computer networks"],
}

FEATURES = ["cgpa", "projects", "internships", "certifications", "skill_match",
            "communication", "backlogs", "college_tier", "career_gap_months"]
LABELS = {
    "cgpa": "CGPA", "projects": "Projects", "internships": "Internships",
    "certifications": "Certifications", "skill_match": "Skill match",
    "communication": "Communication", "backlogs": "Backlogs",
    "college_tier": "College tier", "career_gap_months": "Career gap",
}


def fmt_value(col, v):
    if col == "skill_match":
        return f"{v:.0f}%"
    if col == "career_gap_months":
        return f"{v:g} mo"
    if col == "cgpa":
        return f"{v:.2f}"
    return f"{v:g}"


def extract_skills(text):
    """Find known skills in free text (resume) using keyword matching."""
    t = text.lower()
    found = []
    for skill, words in ALIASES.items():
        for w in words:
            if re.search(r"(?<![a-z0-9+#])" + re.escape(w) + r"(?![a-z0-9+#])", t):
                found.append(skill)
                break
    return sorted(found)


def skill_match_for(role, skills):
    req = ROLES[role]
    matched = [s for s in req if s in skills]
    missing = [s for s in req if s not in skills]
    return 100.0 * len(matched) / len(req), matched, missing


# =====================================================================
# Synthetic data + models
# =====================================================================
def generate_data(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    role = rng.choice(list(ROLES), n)
    gender = rng.choice(["Male", "Female"], n, p=[0.6, 0.4])
    is_f = (gender == "Female").astype(int)
    z = rng.normal(0, 1, n)  # hidden ability

    cgpa = np.clip(7.6 + 0.6 * z + rng.normal(0, 0.7, n), 5, 10).round(2)
    projects = np.clip(rng.poisson(np.exp(0.75 + 0.2 * z)), 0, 6)
    interns = np.clip(rng.poisson(np.exp(-0.2 + 0.3 * z)), 0, 3)
    certs = np.clip(rng.poisson(np.exp(0.3 + 0.2 * z)), 0, 6)
    req = np.array([len(ROLES[r]) for r in role])
    matched = rng.binomial(req, 1 / (1 + np.exp(-(0.8 * z + 0.1))))
    skill = (100 * matched / req).round(1)
    comm = np.clip(rng.normal(6.5 + 0.3 * z, 1.4, n), 1, 10).round(1)
    backlogs = np.clip(rng.poisson(np.exp(-0.7 - 0.4 * z)), 0, 5)
    tier = rng.choice([1, 2, 3], n, p=[0.2, 0.4, 0.4])
    # Proxy feature: career gaps are longer for women in this data
    gap = np.clip(rng.normal(2 + 5 * is_f, 3, n), 0, 24).round(0)

    merit = (0.9 * (cgpa - 7.5) + 0.35 * (projects - 2) + 0.6 * (interns - 1) + 0.15 * (certs - 1)
             + 0.035 * (skill - 50) + 0.2 * (comm - 6.5) - 0.5 * backlogs - 0.5 * (tier - 1))
    # Historical bias baked into past decisions (what the model will try to copy)
    biased = merit - 0.4 * is_f - 0.18 * gap
    hired = (rng.random(n) < 1 / (1 + np.exp(-(biased + 1.0)))).astype(int)

    return pd.DataFrame({"gender": gender, "role": role, "cgpa": cgpa, "projects": projects,
                         "internships": interns, "certifications": certs, "skill_match": skill,
                         "communication": comm, "backlogs": backlogs, "college_tier": tier,
                         "career_gap_months": gap, "hired": hired})


def fair_proba(fair, X):
    try:
        return fair._pmf_predict(X)[:, 1]
    except Exception:
        return fair.predict(X).astype(float)


@st.cache_resource(show_spinner="Training models (first load only, ~30s)...")
def build():
    df = generate_data()
    X, y, g = df[FEATURES], df["hired"], df["gender"]
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

    pred_rf = rf.predict(X_te)
    pred_fair = (fair_proba(fair, X_te) >= 0.5).astype(int)
    audit = pd.DataFrame([report("Before (standard model)", pred_rf),
                          report("After (bias-mitigated)", pred_fair)])

    sv = explainer(X_te)
    vals = sv.values[:, :, 1] if sv.values.ndim == 3 else sv.values
    perf = {"accuracy": accuracy_score(y_te, pred_rf), "f1": f1_score(y_te, pred_rf),
            "precision": precision_score(y_te, pred_rf), "recall": recall_score(y_te, pred_rf)}
    return {"df": df, "X": X, "X_te": X_te, "rf": rf, "fair": fair, "explainer": explainer,
            "audit": audit, "vals": vals, "perf": perf}


def proba(app, model_choice, X):
    if model_choice == "Bias-mitigated model":
        return fair_proba(app["fair"], X)
    return app["rf"].predict_proba(X)[:, 1]


def shap_for(app, row):
    sv = app["explainer"](row)
    vals = sv.values[0, :, 1] if sv.values.ndim == 3 else sv.values[0]
    return vals


# =====================================================================
# Counterfactual: smallest set of realistic changes that flips a rejection
# =====================================================================
def counterfactuals(app, model_choice, base, missing, nreq):
    actions = [(f"Learn {s}", "skill_match", 100.0 / nreq) for s in missing]
    if base["internships"] < 3:
        actions.append(("Complete 1 more internship", "internships", 1))
    if base["projects"] < 6:
        actions.append(("Add 1 more project", "projects", 1))
    if base["certifications"] < 6:
        actions.append(("Earn 1 more certification", "certifications", 1))
    if base["backlogs"] > 0:
        actions.append(("Clear 1 backlog", "backlogs", -1))
    if base["communication"] <= 9:
        actions.append(("Improve communication score by 1 point", "communication", 1))

    combos, rows = [], []
    for k in (1, 2, 3):
        for c in itertools.combinations(range(len(actions)), k):
            r = dict(base)
            for i in c:
                _, col, d = actions[i]
                r[col] = r[col] + d
            r["skill_match"] = min(r["skill_match"], 100.0)
            combos.append(c)
            rows.append(r)
    if not rows:
        return []
    p = proba(app, model_choice, pd.DataFrame(rows)[FEATURES])
    flips = [(c, pr) for c, pr in zip(combos, p) if pr >= 0.5]
    flips.sort(key=lambda t: (len(t[0]), -t[1]))

    minimal = []
    for c, pr in flips:
        if any(set(prev).issubset(set(c)) for prev, _ in minimal):
            continue  # a smaller combination already works
        minimal.append((c, pr))

    # Group options that are equivalent (e.g. "learn React" vs "learn SQL" give the same result)
    groups, seen = [], set()
    for c, pr in minimal:
        skill_names = [actions[i][0].replace("Learn ", "") for i in c if actions[i][1] == "skill_match"]
        others = tuple(actions[i][0] for i in c if actions[i][1] != "skill_match")
        key = (others, len(skill_names), round(float(pr), 2))
        if key in seen:
            continue
        seen.add(key)
        parts = list(others)
        if skill_names:
            k = len(skill_names)
            parts.insert(0, f"Learn {k} more required skill{'s' if k > 1 else ''} (e.g. {', '.join(skill_names)})")
        groups.append((parts, float(pr), len(c)))
        if len(groups) == 3:
            break
    return [(parts, pr) for parts, pr, _ in groups]


# =====================================================================
# Charts (plain matplotlib, white background)
# =====================================================================
def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def contrib_chart(vals, row):
    order = np.argsort(np.abs(vals))
    names = [f"{LABELS[FEATURES[i]]} = {fmt_value(FEATURES[i], row.iloc[0, i])}" for i in order]
    v = vals[order]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh(names, v, color=["#2E9E6B" if x > 0 else "#D64545" for x in v])
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("Pushes toward REJECT (left, red)   |   toward SHORTLIST (right, green)")
    style_ax(ax)
    fig.tight_layout()
    return fig


def beeswarm_chart(vals, X):
    rng = np.random.default_rng(0)
    order = np.argsort(np.abs(vals).mean(0))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sc = None
    for pos, i in enumerate(order):
        col = X.iloc[:, i].values.astype(float)
        span = col.max() - col.min()
        norm = (col - col.min()) / (span if span else 1)
        y = pos + rng.uniform(-0.28, 0.28, len(col))
        sc = ax.scatter(vals[:, i], y, c=norm, cmap="coolwarm", s=9, alpha=0.8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([LABELS[X.columns[i]] for i in order])
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("Impact on decision (right = toward shortlist)")
    fig.colorbar(sc, ax=ax).set_label("Feature value (low to high)")
    style_ax(ax)
    fig.tight_layout()
    return fig


def bar_chart(names, values, color="#1F3C88", xlabel="", figsize=(6, 3.4)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(names, values, color=color)
    ax.set_xlabel(xlabel)
    style_ax(ax)
    fig.tight_layout()
    return fig


# =====================================================================
# Styling
# =====================================================================
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; max-width: 1250px;}
.hero {background: linear-gradient(120deg, #1F3C88 0%, #0F9B8E 100%); color: white;
       padding: 22px 28px; border-radius: 14px; margin-bottom: 14px;}
.hero h1 {color: white; margin: 0; font-size: 2.0rem;}
.hero p {color: #E8F3F2; margin: 4px 0 0 0; font-size: 1.0rem;}
.dcard {border-radius: 12px; padding: 16px 20px; border-left: 8px solid; margin-bottom: 8px;}
.dcard.ok {background: rgba(46,158,107,0.15); border-color: #2E9E6B;}
.dcard.no {background: rgba(214,69,69,0.15); border-color: #D64545;}
.dcard .tag {font-size: 0.75rem; letter-spacing: 1.5px; opacity: 0.75;}
.dcard .big {font-size: 1.7rem; font-weight: 700; line-height: 1.3;}
.chip {display: inline-block; padding: 3px 11px; margin: 2px 4px 2px 0; border-radius: 999px;
       font-size: 0.85rem; border: 1px solid;}
.chip.good {background: rgba(46,158,107,0.15); border-color: #2E9E6B;}
.chip.bad {background: rgba(214,69,69,0.13); border-color: #D64545;}
.chip.neutral {background: rgba(120,120,120,0.13); border-color: #888;}
.step {border-left: 4px solid #0F9B8E; padding: 4px 12px; margin: 6px 0; border-radius: 4px;
       background: rgba(15,155,142,0.08);}
.small-note {font-size: 0.82rem; opacity: 0.75;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>⚖️ FairHire</h1>
  <p>An Explainable and Bias-Aware Hiring Decision Support System &nbsp;|&nbsp; IBM AIML Mini Project, PS 50</p>
</div>
""", unsafe_allow_html=True)

app = build()

# =====================================================================
# Sidebar
# =====================================================================
with st.sidebar:
    st.header("Settings")
    model_choice = st.radio("Decision model", ["Standard model", "Bias-mitigated model"], key="model_choice",
                            help="The bias-mitigated model is trained with a Fairlearn fairness constraint.")
    st.caption("Standard = Random Forest trained on past decisions.  \n"
               "Bias-mitigated = same idea, trained to select men and women at similar rates.")
    st.divider()
    d = app["df"]
    st.metric("Training candidates (synthetic)", f"{len(d):,}")
    st.metric("Job roles supported", len(ROLES))
    st.divider()
    st.caption("⚠️ Decision support only. A human recruiter must make the final decision. "
               "All data is synthetic.")

# =====================================================================
# Presets
# =====================================================================
PRESETS = {
    "Strong ML candidate": dict(role="Machine Learning Engineer",
                                skills=["Python", "scikit-learn", "Machine Learning", "Statistics", "SQL", "Git"],
                                cgpa=8.8, projects=4, internships=2, certs=3, comm=8.0, backlogs=0, tier=1, gap=1),
    "Average web developer": dict(role="Web Developer",
                                  skills=["HTML", "CSS", "JavaScript", "Git"],
                                  cgpa=7.4, projects=2, internships=1, certs=1, comm=6.5, backlogs=0, tier=2, gap=2),
    "Good profile with career gap": dict(role="Data Analyst",
                                         skills=["Python", "SQL", "Excel", "Statistics", "Pandas"],
                                         cgpa=8.2, projects=3, internships=1, certs=2, comm=7.0, backlogs=0, tier=2, gap=12),
}
DEFAULTS = dict(role="Machine Learning Engineer", mode="Select skills", skills=["Python", "SQL"], resume="",
                cgpa=7.5, projects=2, internships=1, certs=1, comm=6.5, backlogs=0, tier=2, gap=0,
                evaluated=False, preset="(none)")
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def apply_preset():
    p = PRESETS.get(st.session_state["preset"])
    if p:
        for k, v in p.items():
            st.session_state[k] = v
        st.session_state["mode"] = "Select skills"
        st.session_state["evaluated"] = False


# =====================================================================
# Tabs
# =====================================================================
t1, t2, t3, t4 = st.tabs(["🧑‍💼 Candidate evaluation", "📊 Model insights",
                          "⚖️ Fairness audit", "🏗️ System architecture"])

# ---------------------------------------------------------------- Tab 1
with t1:
    left, right = st.columns([1, 1.25], gap="large")

    with left:
        st.selectbox("Quick start: load a sample profile", ["(none)"] + list(PRESETS),
                     key="preset", on_change=apply_preset)

        with st.container(border=True):
            st.markdown("**1. Job role and skills**")
            st.selectbox("Applying for", list(ROLES), key="role")
            st.caption("Required: " + ", ".join(ROLES[st.session_state["role"]]))
            st.radio("How do you want to give the candidate's skills?",
                     ["Select skills", "Paste resume text"], key="mode", horizontal=True)
            if st.session_state["mode"] == "Select skills":
                st.multiselect("Candidate's skills", ALL_SKILLS, key="skills")
                skills = st.session_state["skills"]
            else:
                st.text_area("Paste resume text here", key="resume", height=130,
                             placeholder="e.g. Built a churn model using Python, pandas and sklearn. "
                                         "Used SQL and Git. Studied statistics and machine learning.")
                skills = extract_skills(st.session_state["resume"])
                st.caption("Skills detected: " + (", ".join(skills) if skills else "none yet"))

        with st.container(border=True):
            st.markdown("**2. Academics and experience**")
            a, b = st.columns(2)
            with a:
                st.slider("CGPA", 5.0, 10.0, step=0.1, key="cgpa")
                st.slider("Projects", 0, 6, key="projects")
                st.slider("Internships", 0, 3, key="internships")
                st.slider("Certifications", 0, 6, key="certs")
            with b:
                st.slider("Communication (1-10)", 1.0, 10.0, step=0.1, key="comm")
                st.slider("Backlogs", 0, 5, key="backlogs")
                st.selectbox("College tier", [1, 2, 3], key="tier")
                st.slider("Career gap (months)", 0, 24, key="gap")

        if st.button("Evaluate candidate", type="primary", width="stretch"):
            st.session_state["evaluated"] = True

    with right:
        if not st.session_state["evaluated"]:
            st.info("Fill in the candidate details on the left (or load a sample profile) "
                    "and click **Evaluate candidate**.")
        else:
            role = st.session_state["role"]
            sm, matched, missing = skill_match_for(role, skills)
            base = {"cgpa": st.session_state["cgpa"], "projects": st.session_state["projects"],
                    "internships": st.session_state["internships"],
                    "certifications": st.session_state["certs"], "skill_match": sm,
                    "communication": st.session_state["comm"], "backlogs": st.session_state["backlogs"],
                    "college_tier": st.session_state["tier"], "career_gap_months": st.session_state["gap"]}
            row = pd.DataFrame([base])[FEATURES]
            p = float(proba(app, model_choice, row)[0])
            shortlisted = p >= 0.5

            cls, title = ("ok", "✅ Shortlist") if shortlisted else ("no", "❌ Not shortlisted")
            st.markdown(f"""<div class="dcard {cls}"><div class="tag">RECOMMENDATION · {model_choice.upper()}</div>
                <div class="big">{title}</div>
                <div>Shortlist probability: <b>{p*100:.0f}%</b> &nbsp;(threshold 50%)</div></div>""",
                        unsafe_allow_html=True)
            st.progress(min(max(p, 0.0), 1.0))

            m1, m2, m3 = st.columns(3)
            m1.metric("Role", role)
            m2.metric("Skill match", f"{sm:.0f}%")
            m3.metric("Skills matched", f"{len(matched)} of {len(ROLES[role])}")

            st.markdown("**Skills for this role**")
            chips = "".join(f'<span class="chip good">✓ {s}</span>' for s in matched) + \
                    "".join(f'<span class="chip bad">✗ {s}</span>' for s in missing)
            st.markdown(chips, unsafe_allow_html=True)

            st.markdown("#### Why this decision?")
            vals = shap_for(app, row)
            order = np.argsort(vals)
            pos = [i for i in order[::-1] if vals[i] > 0.01][:3]
            neg = [i for i in order if vals[i] < -0.01][:3]
            f = lambda i: f"{LABELS[FEATURES[i]]} ({fmt_value(FEATURES[i], row.iloc[0, i])})"
            st.markdown("**Helping:** " + (", ".join(f(i) for i in pos) if pos else "no strong positives"))
            st.markdown("**Hurting:** " + (", ".join(f(i) for i in neg) if neg else "no strong negatives"))
            st.pyplot(contrib_chart(vals, row), clear_figure=True)
            if model_choice == "Bias-mitigated model":
                st.caption("Explanation chart is computed from the standard model. The bias-mitigated "
                           "model is an ensemble trained under a fairness constraint.")

            if base["career_gap_months"] >= 6:
                st.warning("This candidate has a career gap of 6+ months. In this dataset, career gap is a "
                           "proxy for gender, so it can create indirect bias. See the Fairness audit tab.")

            st.markdown("#### What would change this decision?")
            if shortlisted:
                st.success("This candidate is already shortlisted. Keeping skills current and adding "
                           "experience will only strengthen the profile.")
            else:
                cfs = counterfactuals(app, model_choice, base, missing, len(ROLES[role]))
                if cfs:
                    st.caption("Smallest realistic changes that would move this candidate to the shortlist:")
                    for acts, pr in cfs:
                        st.markdown(f'<div class="step">{" + ".join(acts)} &nbsp;→&nbsp; '
                                    f'<b>{pr*100:.0f}%</b> shortlist probability</div>', unsafe_allow_html=True)
                    st.caption("Only changes the candidate can act on are suggested (career gap and "
                               "college tier are deliberately excluded).")
                else:
                    st.info("No combination of up to 3 realistic changes was enough. "
                            "A larger improvement across several areas is needed.")

# ---------------------------------------------------------------- Tab 2
with t2:
    st.subheader("How the model makes decisions")
    pf = app["perf"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{pf['accuracy']:.3f}")
    c2.metric("F1 score", f"{pf['f1']:.3f}")
    c3.metric("Precision", f"{pf['precision']:.3f}")
    c4.metric("Recall", f"{pf['recall']:.3f}")
    st.caption("Standard Random Forest, evaluated on a held-out 20% test set.")

    ca, cb = st.columns(2)
    with ca:
        st.markdown("**Overall feature importance (SHAP)**")
        imp = pd.Series(np.abs(app["vals"]).mean(0), index=[LABELS[c] for c in FEATURES]).sort_values()
        st.pyplot(bar_chart(imp.index, imp.values, xlabel="Average impact on decision"), clear_figure=True)
    with cb:
        st.markdown("**Shortlist rate by job role (training data)**")
        hr = app["df"].groupby("role").hired.mean().sort_values()
        st.pyplot(bar_chart(hr.index, hr.values, color="#0F9B8E", xlabel="Share of candidates hired"),
                  clear_figure=True)

    st.markdown("**Detailed view: every test candidate, every feature**")
    st.pyplot(beeswarm_chart(app["vals"], app["X_te"]), clear_figure=True)
    st.caption("Each dot is one candidate. Red = high feature value, blue = low. Dots to the right pushed "
               "the decision toward shortlist. For example, high CGPA pushes right, high career gap pushes left.")
    st.info("Career gap appears among the important features. That matters because it is linked to gender "
            "in this dataset (see the Fairness audit tab).")

# ---------------------------------------------------------------- Tab 3
with t3:
    st.subheader("Does the model treat men and women equally?")
    st.write("Gender is **never** an input to the model. It is used only to audit the results afterwards.")
    a = app["audit"]
    b0, b1 = a.iloc[0], a.iloc[1]

    k1, k2, k3 = st.columns(3)
    k1.metric("Selection gap (demographic parity)", f"{b1['Demographic parity diff']:.3f}",
              delta=f"{b1['Demographic parity diff'] - b0['Demographic parity diff']:+.3f} vs before",
              delta_color="inverse")
    k2.metric("Equalized odds gap", f"{b1['Equalized odds diff']:.3f}",
              delta=f"{b1['Equalized odds diff'] - b0['Equalized odds diff']:+.3f} vs before",
              delta_color="inverse")
    k3.metric("Accuracy", f"{b1['Accuracy']:.3f}",
              delta=f"{b1['Accuracy'] - b0['Accuracy']:+.3f} vs before")
    st.caption("Lower gaps are fairer (0 = identical treatment). Mitigation usually costs a little accuracy.")
    st.info("The fix targets the **selection gap** (demographic parity), and that gap drops sharply. "
            "The equalized odds gap changes very little. Different fairness measures can pull in different "
            "directions, so no single mitigation makes a model fair on every definition at once.")

    st.dataframe(a, hide_index=True, width="stretch")

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    x = np.arange(2)
    ax.bar(x - 0.18, a["Female selection rate"], 0.36, label="Female", color="#D64545")
    ax.bar(x + 0.18, a["Male selection rate"], 0.36, label="Male", color="#1F3C88")
    ax.set_xticks(x)
    ax.set_xticklabels(["Before (standard)", "After (bias-mitigated)"])
    ax.set_ylabel("Shortlist rate")
    ax.legend(frameon=False)
    style_ax(ax)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

    st.markdown("**Why was there bias without gender as an input?**")
    gg = app["df"].groupby("gender").career_gap_months.mean().round(1)
    st.write(f"Past decisions in the training data penalised career gaps, and women have longer gaps on "
             f"average (**{gg['Female']}** months vs **{gg['Male']}** for men). The model learned gender "
             f"indirectly through this proxy feature.")
    st.write("**How it was reduced:** Fairlearn's ExponentiatedGradient trains the model under a "
             "demographic-parity constraint, so selection rates for the two groups are pushed together.")

# ---------------------------------------------------------------- Tab 4
with t4:
    st.subheader("System architecture")
    dot = """
    digraph G {
      rankdir=LR; bgcolor="transparent";
      node [shape=box, style="rounded,filled", fillcolor="#EAF1FB", color="#1F3C88",
            fontname="Helvetica", fontsize=11, fontcolor="#10223F"];
      edge [color="#5A6B85"];
      resume [label="Resume text /\\nskill checklist"];
      role   [label="Target job role"];
      skills [label="Skill extraction\\n& matching"];
      acad   [label="Academic &\\nexperience inputs"];
      feat   [label="Feature vector\\n(9 features)", fillcolor="#DDF3F0", color="#0F9B8E"];
      rf     [label="Standard model\\n(Random Forest)"];
      fair   [label="Bias-mitigated model\\n(Fairlearn)"];
      dec    [label="Decision +\\nconfidence", fillcolor="#DDF3F0", color="#0F9B8E"];
      shap   [label="SHAP\\nexplanation"];
      cf     [label="Counterfactual\\nsearch"];
      audit  [label="Fairness audit\\n(by gender)", fillcolor="#FDECEC", color="#D64545"];
      ui     [label="Recruiter dashboard\\n(Streamlit)", fillcolor="#FFF4DB", color="#C58A00"];
      resume -> skills; role -> skills; skills -> feat; acad -> feat;
      feat -> rf; feat -> fair; rf -> dec; fair -> dec;
      rf -> shap; dec -> cf; rf -> audit; fair -> audit;
      dec -> ui; shap -> ui; cf -> ui; audit -> ui;
    }
    """
    st.graphviz_chart(dot, width="stretch")

    st.markdown("**What each stage does**")
    st.markdown("""
| Stage | Purpose |
|---|---|
| Skill extraction and matching | Reads the resume (or checklist) and compares it to the chosen role's required skills |
| Feature vector | Combines skill match with CGPA, projects, internships, certifications, communication, backlogs, college tier and career gap |
| Standard model | Random Forest trained on past (biased) hiring decisions |
| Bias-mitigated model | Random Forest trained under a demographic-parity constraint with Fairlearn |
| SHAP explanation | Shows how much each feature pushed this candidate toward or away from shortlist |
| Counterfactual search | Finds the smallest realistic changes that would flip a rejection |
| Fairness audit | Compares selection rates and error rates for men and women before and after mitigation |
""")

    st.markdown("**Job roles and required skills**")
    roles_df = pd.DataFrame({"Role": list(ROLES), "Required skills": [", ".join(v) for v in ROLES.values()]})
    st.dataframe(roles_df, hide_index=True, width="stretch")

    st.markdown("**Technology stack:** Python, pandas, NumPy, scikit-learn, SHAP, Fairlearn, matplotlib, Streamlit")

    st.markdown("**Limitations**")
    st.markdown("""
- All candidate data is **synthetic**, with bias planted deliberately to demonstrate the method
- Resume reading is keyword matching, so it can miss skills written in unusual ways
- Only one sensitive attribute (gender) and one fairness definition (demographic parity) are checked
- Fairness techniques reduce bias but do not remove it completely, and they can reduce accuracy
- Intended as decision support. Real hiring needs human review and real, audited data
""")
