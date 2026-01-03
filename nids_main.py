# ================== IMPORTS ==================
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="AI-Based Network Intrusion Detection System",
    layout="wide"
)

st.title("AI-Based Network Intrusion Detection System")
st.markdown("""
This project uses **Machine Learning (Random Forest Algorithm)**  
to classify network traffic as **Benign (Normal)** or **Malicious (Attack)**.
""")


# ================== DATA GENERATION ==================
@st.cache_data
def create_dataset():
    np.random.seed(42)
    total_rows = 5000

    data = {
        "Destination_Port": np.random.randint(1, 65535, total_rows),
        "Flow_Duration": np.random.randint(50, 100000, total_rows),
        "Total_Fwd_Packets": np.random.randint(1, 120, total_rows),
        "Packet_Length_Mean": np.random.uniform(20, 1500, total_rows),
        "Active_Mean": np.random.uniform(0, 1000, total_rows),
        "Label": np.random.choice([0, 1], total_rows, p=[0.7, 0.3])
    }

    df = pd.DataFrame(data)

    # Attack pattern simulation
    attack_index = df["Label"] == 1
    df.loc[attack_index, "Total_Fwd_Packets"] += np.random.randint(
        50, 200, attack_index.sum()
    )
    df.loc[attack_index, "Flow_Duration"] = np.random.randint(
        1, 1000, attack_index.sum()
    )

    return df


df = create_dataset()


# ================== SIDEBAR ==================
st.sidebar.header("Model Settings")

train_percentage = st.sidebar.slider("Training Data (%)", 60, 90, 80)
n_trees = st.sidebar.slider("Number of Trees (Random Forest)", 50, 200, 100)


# ================== DATA SPLIT ==================
X = df.drop("Label", axis=1)
y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=(100 - train_percentage) / 100,
    random_state=42
)


# ================== TRAIN MODEL ==================
st.divider()
col_train, col_result = st.columns([1, 2])

with col_train:
    st.subheader("1. Model Training")

    if st.button("Train Model"):
        with st.spinner("Training the model..."):
            model = RandomForestClassifier(
                n_estimators=n_trees,
                random_state=42
            )
            model.fit(X_train, y_train)
            st.session_state["model"] = model

        st.success("Model trained successfully!")

    if "model" in st.session_state:
        st.info("Model is ready for prediction")


# ================== EVALUATION ==================
with col_result:
    st.subheader("2. Performance Metrics")

    if "model" in st.session_state:
        model = st.session_state["model"]
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{accuracy * 100:.2f}%")
        m2.metric("Total Samples", len(df))
        m3.metric("Threats Detected", int(np.sum(y_pred)))

        st.write("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", ax=ax)
        st.pyplot(fig)

    else:
        st.warning("Please train the model first")


# ================== LIVE TRAFFIC TEST ==================
st.divider()
st.subheader("3. Live Network Traffic Analysis")

c1, c2, c3, c4 = st.columns(4)

flow_duration = c1.number_input("Flow Duration", 0, 100000, 500)
packet_count = c2.number_input("Total Forward Packets", 0, 500, 100)
packet_mean = c3.number_input("Packet Length Mean", 0, 1500, 500)
active_time = c4.number_input("Active Mean", 0, 1000, 50)

if st.button("Analyze Traffic"):
    if "model" in st.session_state:
        model = st.session_state["model"]

        input_data = np.array([[
            80,
            flow_duration,
            packet_count,
            packet_mean,
            active_time
        ]])

        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.error("⚠️ MALICIOUS TRAFFIC DETECTED")
            st.write("Reason: Abnormal packet behavior observed.")
        else:
            st.success("✅ BENIGN TRAFFIC (Safe)")
    else:
        st.error("Please train the model first")
