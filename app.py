# ==========================================
# STUDENT MENTAL HEALTH DASHBOARD
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ==========================================
# PAGE CONFIG (AGAR TERLIHAT PROFESIONAL)
# ==========================================

st.set_page_config(
    page_title="Student Mental Health Dashboard",
    page_icon="🧠",
    layout="wide"
)

# ==========================================
# TITLE
# ==========================================

st.title("🧠 Student Mental Health Dashboard")
st.write("Dashboard Analisis dan Prediksi Kesehatan Mental Mahasiswa menggunakan Machine Learning")

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("Student Mental health (1).csv")

# Rename kolom
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

# Cleaning
df = df.dropna()

# Encoding
df['depression'] = df['depression'].map({'Yes':1,'No':0})
df['anxiety'] = df['anxiety'].map({'Yes':1,'No':0})
df['panic_attack'] = df['panic_attack'].map({'Yes':1,'No':0})
df['treatment'] = df['treatment'].map({'Yes':1,'No':0})

# Feature engineering
df["mental_health_score"] = df["depression"] + df["anxiety"] + df["panic_attack"]

# ==========================================
# SIDEBAR MENU
# ==========================================

menu = st.sidebar.selectbox(
    "Menu",
    ["Dashboard Data","Visualisasi Data","Machine Learning","Prediksi"]
)

# ==========================================
# DASHBOARD DATA
# ==========================================

if menu == "Dashboard Data":

    st.header("📊 Dataset Overview")

    st.write("Jumlah Data:", df.shape)

    st.dataframe(df.head(10))

    st.subheader("Statistik Dataset")

    st.write(df.describe())

# ==========================================
# VISUALISASI DATA
# ==========================================

elif menu == "Visualisasi Data":

    st.header("📈 Visualisasi Data Mental Health")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Distribusi Gender")

        fig1, ax1 = plt.subplots()

        sns.countplot(x='gender', data=df, ax=ax1)

        st.pyplot(fig1)

    with col2:

        st.subheader("Mahasiswa Mengalami Depresi")

        fig2, ax2 = plt.subplots()

        sns.countplot(x='depression', data=df, ax=ax2)

        st.pyplot(fig2)

    col3, col4 = st.columns(2)

    with col3:

        st.subheader("Mahasiswa Mengalami Anxiety")

        fig3, ax3 = plt.subplots()

        sns.countplot(x='anxiety', data=df, ax=ax3)

        st.pyplot(fig3)

    with col4:

        st.subheader("Mahasiswa Mengalami Panic Attack")

        fig4, ax4 = plt.subplots()

        sns.countplot(x='panic_attack', data=df, ax=ax4)

        st.pyplot(fig4)

    st.subheader("Heatmap Korelasi")

    fig5, ax5 = plt.subplots(figsize=(8,6))

    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax5)

    st.pyplot(fig5)

# ==========================================
# MACHINE LEARNING
# ==========================================

elif menu == "Machine Learning":

    st.header("🤖 Machine Learning Model")

    X = df[['age','mental_health_score']]
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

    st.subheader("Model Logistic Regression")

    st.write("Model Accuracy:", round(accuracy*100,2), "%")

# ==========================================
# PREDIKSI
# ==========================================

elif menu == "Prediksi":

    st.header("🔮 Prediksi Depression Mahasiswa")

    X = df[['age','mental_health_score']]
    y = df['depression']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LogisticRegression()

    model.fit(X_train, y_train)

    age = st.number_input("Masukkan Umur Mahasiswa", 17, 40)

    mental_score = st.slider("Mental Health Score", 0, 3)

    if st.button("Prediksi"):

        prediction = model.predict([[age, mental_score]])

        if prediction[0] == 1:

            st.error("⚠️ Mahasiswa Berisiko Mengalami Depression")

        else:

            st.success("✅ Mahasiswa Tidak Mengalami Depression")

# ==========================================
# FOOTER
# ==========================================

st.sidebar.write("---")
st.sidebar.write("Project Machine Learning")
st.sidebar.write("Student Mental Health Analysis")
