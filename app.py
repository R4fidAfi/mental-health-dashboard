import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="Mental Health AI Dashboard",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Student Mental Health AI Dashboard")
st.markdown("Dashboard analisis kesehatan mental mahasiswa menggunakan **Machine Learning**")

# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv("student-mental-health.csv")

df.rename(columns={
    "Choose your gender": "gender",
    "Age": "age",
    "What is your course?": "course",
    "Your current year of Study": "year",
    "What is your CGPA?": "cgpa",
    "Marital status": "marital_status",
    "Do you have Depression?": "depression",
    "Do you have Anxiety?": "anxiety",
    "Do you have Panic attack?": "panic_attack",
    "Did you seek any specialist for a treatment?": "treatment"
}, inplace=True)

df = df.dropna()

df['depression'] = df['depression'].map({'Yes':1,'No':0})
df['anxiety'] = df['anxiety'].map({'Yes':1,'No':0})
df['panic_attack'] = df['panic_attack'].map({'Yes':1,'No':0})
df['treatment'] = df['treatment'].map({'Yes':1,'No':0})

df["mental_score"] = df["depression"] + df["anxiety"] + df["panic_attack"]

# ==============================
# SIDEBAR
# ==============================

st.sidebar.title("Menu Dashboard")

menu = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Visualisasi", "AI Prediction"]
)

# ==============================
# DASHBOARD
# ==============================

if menu == "Dashboard":

    st.subheader("📊 Statistik Dataset")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Data", len(df))
    col2.metric("Depression Cases", df['depression'].sum())
    col3.metric("Anxiety Cases", df['anxiety'].sum())
    col4.metric("Panic Attack Cases", df['panic_attack'].sum())

    st.divider()

    col5, col6 = st.columns(2)

    with col5:

        gender_chart = px.histogram(
            df,
            x="gender",
            title="Distribusi Gender",
            color="gender"
        )

        st.plotly_chart(gender_chart, use_container_width=True)

    with col6:

        year_chart = px.histogram(
            df,
            x="year",
            title="Distribusi Tahun Studi",
            color="year"
        )

        st.plotly_chart(year_chart, use_container_width=True)

# ==============================
# VISUALISASI
# ==============================

elif menu == "Visualisasi":

    st.subheader("📈 Analisis Kesehatan Mental")

    col1, col2 = st.columns(2)

    with col1:

        dep_chart = px.pie(
            df,
            names="depression",
            title="Depression Distribution"
        )

        st.plotly_chart(dep_chart, use_container_width=True)

    with col2:

        anx_chart = px.pie(
            df,
            names="anxiety",
            title="Anxiety Distribution"
        )

        st.plotly_chart(anx_chart, use_container_width=True)

    panic_chart = px.histogram(
        df,
        x="mental_score",
        title="Mental Health Score Distribution",
        color="mental_score"
    )

    st.plotly_chart(panic_chart, use_container_width=True)

    cgpa_chart = px.scatter(
        df,
        x="age",
        y="cgpa",
        color="depression",
        title="Age vs CGPA vs Depression"
    )

    st.plotly_chart(cgpa_chart, use_container_width=True)

# ==============================
# MACHINE LEARNING
# ==============================

elif menu == "AI Prediction":

    st.subheader("🤖 AI Prediction Panel")

    X = df[['age','mental_score']]
    y = df['depression']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LogisticRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    st.success(f"Model Accuracy : {round(accuracy*100,2)} %")

    st.divider()

    st.subheader("Input Data Mahasiswa")

    age = st.slider("Age", 17, 40, 21)
    mental_score = st.slider("Mental Health Score", 0, 3)

    if st.button("Predict"):

        result = model.predict([[age, mental_score]])

        if result[0] == 1:

            st.error("⚠️ Berisiko Mengalami Depression")

        else:

            st.success("✅ Tidak Mengalami Depression")

# ==============================
# FOOTER
# ==============================

st.sidebar.markdown("---")
st.sidebar.write("Machine Learning Project")
st.sidebar.write("Student Mental Health Analysis")
