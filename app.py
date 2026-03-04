import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import plotly.express as px

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

st.title("🛍️ Customer Segmentation Dashboard")

st.write("Analyze customer groups based on **Annual Income** and **Spending Score** using K-Means clustering.")

# Load dataset
data = pd.read_csv("Mall_Customers.csv")

# KPI Metrics
st.subheader("Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", data.shape[0])
col2.metric("Average Income", round(data["Annual Income (k$)"].mean(),2))
col3.metric("Average Spending Score", round(data["Spending Score (1-100)"].mean(),2))

st.divider()

# Dataset Explorer
st.subheader("Dataset Preview")

if st.checkbox("Show Dataset"):
    st.dataframe(data)

# Distribution Charts
st.subheader("Customer Distributions")

col1, col2 = st.columns(2)

fig1 = px.histogram(data, x="Age", title="Age Distribution")
fig2 = px.histogram(data, x="Annual Income (k$)", title="Income Distribution")

col1.plotly_chart(fig1, use_container_width=True)
col2.plotly_chart(fig2, use_container_width=True)

fig3 = px.histogram(data, x="Spending Score (1-100)", title="Spending Score Distribution")
st.plotly_chart(fig3, use_container_width=True)

st.divider()

# Feature Selection
X = data.iloc[:, [3,4]].values

# Elbow Method
st.subheader("Elbow Method")

wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

fig = px.line(
    x=range(1,11),
    y=wcss,
    markers=True,
    labels={"x":"Number of Clusters","y":"WCSS"},
    title="Optimal Clusters using Elbow Method"
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# Train Model
kmeans = KMeans(n_clusters=5, init="k-means++", random_state=42)
clusters = kmeans.fit_predict(X)

data["Cluster"] = clusters

# Cluster Visualization
st.subheader("Customer Segments")

fig2 = px.scatter(
    data,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    color=data["Cluster"].astype(str),
    title="Customer Segments",
    hover_data=["Age","Gender"]
)

st.plotly_chart(fig2, use_container_width=True)

st.divider()

# Cluster Insights
st.subheader("Cluster Insights")

cluster_summary = data.groupby("Cluster")[["Annual Income (k$)","Spending Score (1-100)"]].mean().round(2)

st.dataframe(cluster_summary)

st.divider()

# Customer Prediction
st.subheader("Predict Customer Segment")

income = st.slider("Annual Income (k$)",10,150,60)
score = st.slider("Spending Score",1,100,50)

if st.button("Predict Segment"):

    prediction = kmeans.predict([[income,score]])

    st.success(f"This customer belongs to **Cluster {prediction[0]}**")

st.divider()

# Download Data
st.subheader("Download Clustered Dataset")

csv = data.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download Segmented Data",
    csv,
    "customer_segments.csv",
    "text/csv"
)
