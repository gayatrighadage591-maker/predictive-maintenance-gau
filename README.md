# Predictive Maintenance – Streamlit Project

Files
- ai4i2020.csv  -> download yourself (UCI / Kaggle "AI4I 2020 Predictive Maintenance Dataset")
- train_model.py (or the .ipynb notebook) -> creates the model files + metrics.json
- app.py -> Streamlit website
- requirements.txt

Run locally
    pip install -r requirements.txt
    python train_model.py
    streamlit run app.py

Deploy: push all files (app.py, requirements.txt, 4 pkl/csv/json outputs) to GitHub,
then share.streamlit.io -> New app -> select repo -> main file: app.py
