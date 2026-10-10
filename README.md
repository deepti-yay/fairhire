# FairHire
An Explainable and Bias-Aware Hiring Decision Support System (IBM AIML PS 50)

Live demo:https://fairhire-ckrun3e5r8yfacqwp36bf2.streamlit.app/

## What it does
- Predicts whether a candidate should be shortlisted
- Explains each decision using SHAP
- Audits gender bias using Fairlearn and reduces it with a mitigation step

## Results
| | Before | After |
|---|---|---|
| Accuracy | 0.757 | 0.733 |
| Demographic parity difference | 0.191 | 0.025 |
| Equalized odds difference | 0.129 | 0.127 |
## Tech
Python, scikit-learn, SHAP, Fairlearn, Streamlit

## Run locally
pip install -r requirements.txt
streamlit run app.py

## Note
Data is synthetic, with bias planted deliberately to demonstrate the method.
