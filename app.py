import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Predictive Maintenance System", page_icon="⚙️")

# ---------------- Team details: EDIT THESE ----------------
GUIDE = "Prof. Aparna Kulkarni"
CLASS_NAME = "TE B"
TEAM = [
    {"Name": "Gayatri Ghadage", "Roll No": "18"},
    {"Name": "Member 2", "Roll No": "__"},
    {"Name": "Member 3", "Roll No": "__"},
]
# -----------------------------------------------------------

TARGET = "Machine failure"
FULL_TITLE = ("Machine Learning-Based Predictive Maintenance and "
              "Machine Failure Risk Prediction System")
UCI_LINK = "https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset"


@st.cache_data
def load_data():
    return pd.read_csv("prepared_predictive_maintenance.csv")


@st.cache_resource
def load_models():
    with open("logistic_regression_model.pkl", "rb") as f:
        lr = pickle.load(f)
    with open("random_forest_model.pkl", "rb") as f:
        rf = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f:
        cols = pickle.load(f)
    return {"Logistic Regression": lr, "Random Forest": rf}, cols


@st.cache_data
def load_metrics():
    with open("metrics.json") as f:
        return json.load(f)


df = load_data()
models, feature_cols = load_models()
metrics = load_metrics()

# ---------------- Sidebar ----------------
st.sidebar.title("⚙️ Predictive Maintenance")
page = st.sidebar.radio("Navigate", [
    "Home", "Dataset Overview", "Data Visualization",
    "Failure Prediction", "Model Comparison", "About Project"])
st.sidebar.divider()
st.sidebar.caption("Machine Learning Mini Project")
st.sidebar.divider()
st.sidebar.caption("Predictive Maintenance ML Project")


# ---------------- Home ----------------
if page == "Home":
    st.title("⚙️ Predictive Maintenance System")
    st.subheader(FULL_TITLE)
    st.write("This project uses machine learning to analyze machine operating "
             "conditions and predict potential machine failure.")
    st.info("Enter machine operating parameters to obtain a prediction from the "
            "trained machine learning model.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Dataset Records", f"{len(df):,}")
    c2.metric("Input Features", len(feature_cols))
    c3.metric("ML Algorithms", len(models))
    st.subheader("Project Objectives")
    st.markdown(
        "- Analyze machine operating conditions.\n"
        "- Explore machine failure patterns.\n"
        "- Train Logistic Regression and Random Forest models.\n"
        "- Compare model performance using classification metrics.\n"
        "- Provide a web application for machine failure prediction.")


# ---------------- Dataset Overview ----------------
elif page == "Dataset Overview":
    st.title("📊 Dataset Overview")
    st.caption("Dataset: AI4I 2020 Predictive Maintenance Dataset")
    c1, c2, c3 = st.columns(3)
    c1.metric("Records", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", int(df.isnull().sum().sum()))
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))
    st.subheader("Statistical Summary")
    st.dataframe(df.describe())
    st.subheader("Column Information")
    st.dataframe(pd.DataFrame({"Data Type": df.dtypes.astype(str)}))


# ---------------- Data Visualization ----------------
elif page == "Data Visualization":
    st.title("📈 Data Visualization")

    st.subheader("Machine Failure Distribution")
    counts = df[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots()
    ax.bar(["No Failure", "Failure"], [counts.get(0, 0), counts.get(1, 0)])
    ax.set_title("Machine Failure Distribution")
    ax.set_ylabel("Number of Records")
    st.pyplot(fig)

    st.subheader("Operating Parameter Distribution")
    param = st.selectbox("Select a parameter", feature_cols)
    fig, ax = plt.subplots()
    ax.hist(df[param], bins=30)
    ax.set_title(f"Distribution of {param}")
    ax.set_xlabel(param)
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    st.pyplot(fig)


# ---------------- Failure Prediction ----------------
elif page == "Failure Prediction":
    st.title("🤖 Machine Failure Prediction")
    st.write("Enter machine operating parameters to predict whether machine "
             "failure is likely.")
    st.warning("Predictions are based on the trained model and are not a "
               "guarantee of actual machine behavior.")

    with st.form("prediction_form"):
        left, right = st.columns(2)
        air = left.number_input("Air Temperature (K)", 290.0, 310.0, 303.2,
                                step=0.1, format="%.2f")
        proc = left.number_input("Process Temperature (K)", 300.0, 320.0, 313.0,
                                 step=0.1, format="%.2f")
        rpm = left.number_input("Rotational Speed (rpm)", 1000, 3000, 1400, step=1)
        torque = right.number_input("Torque (Nm)", 0.0, 100.0, 60.0,
                                    step=0.1, format="%.2f")
        wear = right.number_input("Tool Wear (min)", 0, 300, 240, step=1)
        model_name = right.selectbox("Choose Prediction Model",
                                     ["Random Forest", "Logistic Regression"])
        submitted = st.form_submit_button("Predict Machine Failure",
                                          use_container_width=True)

    if submitted:
        X = pd.DataFrame([[air, proc, rpm, torque, wear]], columns=feature_cols)
        model = models[model_name]
        pred = int(model.predict(X)[0])
        prob = float(model.predict_proba(X)[0][1])
        st.subheader("Prediction Result")
        if pred == 1:
            st.error("⚠️ Machine Failure Predicted")
        else:
            st.success("✅ No Machine Failure Predicted")
        st.metric("Estimated Failure Probability", f"{prob * 100:.1f}%")


# ---------------- Model Comparison ----------------
elif page == "Model Comparison":
    st.title("🏆 Model Comparison")
    names = [m for m in metrics if not m.startswith("_")]
    table = pd.DataFrame({
        m: {k: v for k, v in metrics[m].items() if k != "Confusion Matrix"}
        for m in names}).T
    st.subheader("Performance Metrics")
    st.dataframe(table.style.format("{:.4f}"))
    st.subheader("Metric Comparison")
    st.bar_chart(table)
    st.success(f"Best model (highest F1-Score): {metrics['_best_model']}")
    st.subheader("Confusion Matrices")
    for m in names:
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(np.array(metrics[m]["Confusion Matrix"]), annot=True, fmt="d",
                    cmap="Blues", ax=ax,
                    xticklabels=["No Failure", "Failure"],
                    yticklabels=["No Failure", "Failure"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(m)
        st.pyplot(fig)
    st.caption("Failures are rare in this dataset, so Recall and F1-Score are "
               "more informative than Accuracy alone.")


# ---------------- About Project ----------------
else:
    st.title("ℹ️ About the Project")
    st.header("Problem Statement")
    st.write("Unexpected machine failures can interrupt industrial operations. "
             "This project explores machine operating data to predict recorded "
             "machine failures using classification.")
    st.header("Methodology")
    st.markdown(
        "1. Dataset collection.\n"
        "2. Data inspection and preprocessing.\n"
        "3. Exploratory data analysis.\n"
        "4. Training Logistic Regression and Random Forest.\n"
        "5. Model evaluation and comparison.\n"
        "6. Deployment through Streamlit.")
    st.header("Applications")
    st.markdown(
        "- Industrial equipment monitoring.\n"
        "- Maintenance planning support.\n"
        "- Analysis of machine operating conditions.\n"
        "- Educational demonstration of predictive maintenance.")
    st.header("Limitations")
    st.markdown(
        "- Predictions depend on the quality and limitations of the dataset.\n"
        "- The application uses manually entered parameters.\n"
        "- The model does not receive live sensor readings.\n"
        "- Predictions require validation before real industrial use.")
    st.header("Future Scope")
    st.markdown(
        "- Integrate real-time sensor data.\n"
        "- Evaluate additional machine learning algorithms.\n"
        "- Explore failure-type prediction.\n"
        "- Improve model calibration and monitoring.")
    st.header("Dataset Source")
    st.markdown(f"[AI4I 2020 Predictive Maintenance Dataset — UCI Repository]({UCI_LINK})")
    st.header("Team Details")
    st.table(pd.DataFrame(TEAM))
    st.write(f"**Class:** {CLASS_NAME}")
    st.write(f"**Project Guide:** {GUIDE}")
