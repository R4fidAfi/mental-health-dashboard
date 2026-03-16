import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# =============================
# PAGE CONFIG
# =============================

st.set_page_config(
    page_title="Mental Health AI Dashboard",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Student Mental Health AI Dashboard")
st.markdown("Advanced Dashboard untuk Analisis Kesehatan Mental Mahasiswa")

# =============================
# LOADING ANIMATION
# =============================

with st.spinner("Loading dataset..."):
    time.sleep(2)

# =============================
# LOAD DATA
# =============================

df = pd.read_csv("Student Mental health (1).csv")

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

df["mental_score"] = df["depression"] + df["anxiety"] + df["panic_attack"]

# =============================
# SIDEBAR
# =============================

st.sidebar.title("Navigation")

menu = st.sidebar.radio(
    "Menu",
    ["Dashboard","Visualization","AI Model Comparison","Prediction"]
)

# =============================
# DASHBOARD
# =============================

if menu == "Dashboard":

    st.subheader("Dataset Statistics")

    col1,col2,col3,col4 = st.columns(4)

    col1.metric("Total Students", len(df))
    col2.metric("Depression Cases", df["depression"].sum())
    col3.metric("Anxiety Cases", df["anxiety"].sum())
    col4.metric("Panic Attack Cases", df["panic_attack"].sum())

    st.divider()

    col5,col6 = st.columns(2)

    with col5:

        fig = px.histogram(df,x="gender",title="Gender Distribution")
        st.plotly_chart(fig,use_container_width=True)

    with col6:

        fig2 = px.histogram(df,x="year",title="Year of Study Distribution")
        st.plotly_chart(fig2,use_container_width=True)

# =============================
# VISUALIZATION
# =============================

elif menu == "Visualization":

    st.subheader("Mental Health Data Analysis")

    col1,col2 = st.columns(2)

    with col1:

        fig = px.pie(df,names="depression",title="Depression Distribution")
        st.plotly_chart(fig,use_container_width=True)

    with col2:

        fig = px.pie(df,names="anxiety",title="Anxiety Distribution")
        st.plotly_chart(fig,use_container_width=True)

    st.subheader("Correlation Heatmap")

    corr = df.corr(numeric_only=True)

    fig, ax = plt.subplots()

    sns.heatmap(corr,annot=True,cmap="coolwarm",ax=ax)

    st.pyplot(fig)

# =============================
# AI MODEL COMPARISON
# =============================

elif menu == "AI Model Comparison":

    st.subheader("AI Algorithm Comparison")

    X = df[["age","mental_score"]]
    y = df["depression"]

    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.2,random_state=42
    )

    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC()
    }

    results = []

    for name,model in models.items():

        model.fit(X_train,y_train)

        pred = model.predict(X_test)

        acc = accuracy_score(y_test,pred)

        results.append([name,acc])

    results_df = pd.DataFrame(results,columns=["Model","Accuracy"])

    st.dataframe(results_df)

    fig = px.bar(
        results_df,
        x="Model",
        y="Accuracy",
        title="Model Accuracy Comparison"
    )

    st.plotly_chart(fig,use_container_width=True)

# =============================
# PREDICTION PANEL
# =============================

elif menu == "Prediction":

    st.subheader("Mental Health Risk Prediction")

    X = df[["age","mental_score"]]
    y = df["depression"]

    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=0.2,random_state=42
    )

    model = RandomForestClassifier()

    model.fit(X_train,y_train)

    age = st.slider("Age",17,40,21)
    mental_score = st.slider("Mental Health Score",0,3)

    if st.button("Predict Risk"):

        pred = model.predict([[age,mental_score]])

        risk = mental_score / 3 * 100

        st.write("Risk Score:", round(risk,2),"%")

        if pred[0]==1:

            st.error("High Risk of Depression")

        elif risk > 40:

            st.warning("Medium Risk")

        else:

            st.success("Low Risk")

# =============================
# AUTO INSIGHT GENERATOR
# =============================

st.sidebar.divider()
st.sidebar.subheader("Auto Insight")

highest_age = df.groupby("age")["depression"].mean().idxmax()

st.sidebar.write(
f"Students age **{highest_age}** memiliki tingkat depresi tertinggi dalam dataset."
)
