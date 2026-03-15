# =====================================
# STUDENT MENTAL HEALTH DASHBOARD
# =====================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# =====================================
# Load Dataset
# =====================================

st.title("🎓 Student Mental Health Dashboard")

st.write("Dashboard Analisis dan Prediksi Kesehatan Mental Mahasiswa")

df = pd.read_csv("Student Mental health (1).csv")


# =====================================
# Rename Kolom
# =====================================

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

df['depression'] = df['depression'].map({'Yes':1, 'No':0})
df['anxiety'] = df['anxiety'].map({'Yes':1, 'No':0})
df['panic_attack'] = df['panic_attack'].map({'Yes':1, 'No':0})


df["mental_health_score"] = df["depression"] + df["anxiety"] + df["panic_attack"]


# =====================================
# Dashboard Dataset
# =====================================

st.header("📊 Dataset Overview")

st.write("Jumlah Data:", df.shape)

st.dataframe(df.head())


# =====================================
# Grafik Gender
# =====================================

st.subheader("Distribusi Gender")

fig1, ax1 = plt.subplots()

sns.countplot(x='gender', data=df, ax=ax1)

st.pyplot(fig1)


# =====================================
# Grafik Depression
# =====================================

st.subheader("Mahasiswa Mengalami Depresi")

fig2, ax2 = plt.subplots()

sns.countplot(x='depression', data=df, ax=ax2)

st.pyplot(fig2)


# =====================================
# Grafik Anxiety
# =====================================

st.subheader("Mahasiswa Mengalami Anxiety")

fig3, ax3 = plt.subplots()

sns.countplot(x='anxiety', data=df, ax=ax3)

st.pyplot(fig3)


# =====================================
# Heatmap Korelasi
# =====================================

st.subheader("Heatmap Korelasi")

fig4, ax4 = plt.subplots()

sns.heatmap(df.corr(numeric_only=True), annot=True, ax=ax4)

st.pyplot(fig4)


# =====================================
# Machine Learning
# =====================================

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

st.write("Model Accuracy:", accuracy)


# =====================================
# Form Prediksi
# =====================================

st.header("🔮 Prediksi Depression Mahasiswa")

age = st.number_input("Umur Mahasiswa", 17, 40)

mental_score = st.slider("Mental Health Score", 0, 3)

if st.button("Prediksi"):

    prediction = model.predict([[age, mental_score]])

    if prediction[0] == 1:

        st.error("⚠️ Mahasiswa Berisiko Depression")

    else:

        st.success("✅ Mahasiswa Tidak Depression")