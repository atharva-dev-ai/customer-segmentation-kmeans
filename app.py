import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px

st.set_page_config(page_title="Customer Segmentation Analytics", layout="wide")

# Load data
data = pd.read_csv("Mall_Customers.csv")

st.sidebar.title("Customer Analytics")

page = st.sidebar.radio(
    "Navigation",
    ["Overview","Data Explorer","Segmentation","Customer Insights","Predict Segment"]
)

# ---------------- OVERVIEW ----------------

if page == "Overview":

    st.title("Customer Segmentation Dashboard")

    col1,col2,col3 = st.columns(3)

    col1.metric("Total Customers",data.shape[0])
    col2.metric("Average Income",round(data["Annual Income (k$)"].mean(),2))
    col3.metric("Average Spending Score",round(data["Spending Score (1-100)"].mean(),2))

    st.subheader("Customer Distribution")

    fig = px.scatter(
        data,
        x="Age",
        y="Spending Score (1-100)",
        color="Gender",
        size="Annual Income (k$)"
    )

    st.plotly_chart(fig,use_container_width=True)

# ---------------- DATA EXPLORER ----------------

elif page == "Data Explorer":

    st.title("Dataset Explorer")

    st.dataframe(data)

    st.subheader("Income Distribution")

    fig = px.histogram(data,x="Annual Income (k$)")
    st.plotly_chart(fig,use_container_width=True)

    st.subheader("Spending Score Distribution")

    fig = px.histogram(data,x="Spending Score (1-100)")
    st.plotly_chart(fig,use_container_width=True)

# ---------------- SEGMENTATION ----------------

elif page == "Segmentation":

    st.title("Customer Segmentation")

    clusters = st.slider("Select Number of Clusters",2,10,5)

    X = data.iloc[:,[3,4]].values

    model = KMeans(n_clusters=clusters,random_state=42)
    labels = model.fit_predict(X)

    data["Cluster"] = labels

    fig = px.scatter(
        data,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        color=data["Cluster"].astype(str)
    )

    st.plotly_chart(fig,use_container_width=True)

    st.subheader("3D Customer Segmentation")

    fig3d = px.scatter_3d(
        data,
        x="Age",
        y="Annual Income (k$)",
        z="Spending Score (1-100)",
        color=data["Cluster"].astype(str)
    )

    st.plotly_chart(fig3d,use_container_width=True)

# ---------------- INSIGHTS ----------------

elif page == "Customer Insights":

    st.title("Cluster Insights")

    X = data.iloc[:,[3,4]].values
    model = KMeans(n_clusters=5,random_state=42)
    data["Cluster"] = model.fit_predict(X)

    summary = data.groupby("Cluster").mean()

    st.dataframe(summary)

    st.subheader("Cluster Sizes")

    fig = px.bar(
        data["Cluster"].value_counts().reset_index(),
        x="index",
        y="Cluster",
        labels={"index":"Cluster","Cluster":"Customers"}
    )

    st.plotly_chart(fig,use_container_width=True)

# ---------------- PREDICTION ----------------

elif page == "Predict Segment":

    st.title("Customer Segment Predictor")

    income = st.slider("Annual Income",10,150,60)
    score = st.slider("Spending Score",1,100,50)

    X = data.iloc[:,[3,4]].values
    model = KMeans(n_clusters=5,random_state=42)
    model.fit(X)

    if st.button("Predict Customer Type"):

        prediction = model.predict([[income,score]])

        st.success(f"This customer belongs to Cluster {prediction[0]}")
