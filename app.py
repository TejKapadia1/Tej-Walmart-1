import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans

from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="Consumer Survey Analytics Dashboard", layout="wide")
st.title("Consumer Survey Analytics Dashboard")

# --- Data Load/Upload Section ---
def load_data():
    uploaded_file = st.sidebar.file_uploader("Upload your CSV data", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("Data uploaded successfully!")
        return df
    elif os.path.exists("data/synthetic_consumer_survey.csv"):
        df = pd.read_csv("data/synthetic_consumer_survey.csv")
        st.sidebar.info("Loaded sample data from data/synthetic_consumer_survey.csv")
        return df
    else:
        st.warning("No data file found. Please upload a CSV file to proceed.")
        return None

df = load_data()
if df is None:
    st.stop()

st.sidebar.write(f"Data shape: {df.shape}")

# --- Tab Navigation ---
tab = st.sidebar.radio(
    "Navigate", 
    [
        "Data Visualization",
        "Classification",
        "Clustering",
        "Association Rule Mining"
    ]
)

# --- Tab 1: Data Visualization ---
if tab == "Data Visualization":
    st.header("Data Visualization & Insights")
    st.write("Explore key patterns and descriptive analytics.")

    # Sidebar filters
    with st.expander("Filter Data"):
        gender = st.multiselect("Gender", df["Gender"].unique())
        age = st.multiselect("Age Group", df["Age"].unique())

    df_vis = df.copy()
    if gender:
        df_vis = df_vis[df_vis["Gender"].isin(gender)]
    if age:
        df_vis = df_vis[df_vis["Age"].isin(age)]

    st.subheader("1. Age Distribution")
    fig1 = px.histogram(df_vis, x="Age", color="Gender", barmode="group")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("2. Income by Age Group")
    fig2 = px.histogram(df_vis, x="Income", color="Age")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("3. Online Shopping Frequency")
    fig3 = px.histogram(df_vis, x="Online_Shop_Frequency", color="Gender")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("4. Distribution of Satisfaction Scores")
    fig4 = px.histogram(df_vis, x="Satisfaction_1_10", nbins=10, color="Gender")
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("5. Main Purchase Factors")
    fig5 = px.histogram(df_vis, x="Main_Purchase_Factor", color="Gender")
    st.plotly_chart(fig5, use_container_width=True)

    st.subheader("6. Frequent Categories (Association Ready)")
    df_cats = df_vis["Frequent_Categories"].str.get_dummies(sep=",")
    st.bar_chart(df_cats.sum())

    st.subheader("7. Price Sensitivity by Age")
    st.plotly_chart(px.histogram(df_vis, x="Price_Sensitivity", color="Age"))

    st.subheader("8. Preferred Shop Time")
    st.plotly_chart(px.histogram(df_vis, x="Preferred_Shop_Time", color="Gender"))

    st.subheader("9. Download Current View")
    st.download_button("Download Filtered Data", df_vis.to_csv(index=False), file_name="filtered_data.csv")

    st.write("More insights can be added based on use-case.")

# --- Tab 2: Classification ---
elif tab == "Classification":
    st.header("Classification: Predicting Willingness to Try New Platform")

    target = "Try_New_Platform"
    features = [
        "Age", "Gender", "Income", "Education", "Employment", "Online_Shop_Frequency",
        "Avg_Spend_Per_Order", "Device_Used", "Main_Purchase_Factor", "Price_Sensitivity"
    ]
    df_class = df[features + [target]].dropna().copy()

    le_dict = {}
    df_encoded = df_class.copy()
    for col in df_encoded.columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        le_dict[col] = le

    X = df_encoded[features]
    y = df_encoded[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "GBRT": GradientBoostingClassifier()
    }
    results = []
    preds_dict = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        preds_dict[name] = preds
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, average="weighted"),
            "Recall": recall_score(y_test, preds, average="weighted"),
            "F1": f1_score(y_test, preds, average="weighted")
        })
    st.subheader("Model Performance")
    st.dataframe(pd.DataFrame(results).round(3))

    st.subheader("Confusion Matrix")
    sel_model = st.selectbox("Select model", list(models.keys()))
    cm = confusion_matrix(y_test, preds_dict[sel_model])
    st.write("Rows: True, Columns: Predicted")
    st.dataframe(cm)

    st.subheader("ROC Curves (All Models)")
    import plotly.graph_objects as go
    fig = go.Figure()
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)
            if y_score.shape[1] > 1:  # handle multiclass
                fpr, tpr, _ = roc_curve(y_test, y_score[:, 1], pos_label=1)
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=name))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), showlegend=False))
    fig.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', width=700, height=500)
    st.plotly_chart(fig)

    st.subheader("Predict on New Uploaded Data")
    uploaded_pred = st.file_uploader("Upload new data (same columns, no target)", key="clf_upload")
    if uploaded_pred:
        df_new = pd.read_csv(uploaded_pred)
        df_new_encoded = df_new.copy()
        for col in features:
            df_new_encoded[col] = le_dict[col].transform(df_new[col].astype(str))
        pred = models[sel_model].predict(df_new_encoded)
        df_new["Predicted_Willingness"] = le_dict[target].inverse_transform(pred)
        st.dataframe(df_new)
        st.download_button("Download Predictions", df_new.to_csv(index=False), file_name="predictions.csv")

# --- Tab 3: Clustering ---
elif tab == "Clustering":
    st.header("Clustering: Customer Segmentation (K-Means)")

    features = [
        "Age", "Gender", "Income", "Education", "Employment", "Online_Shop_Frequency",
        "Avg_Spend_Per_Order", "Device_Used"
    ]
    df_clust = df[features].dropna().copy()
    df_encoded = df_clust.copy()
    for col in features:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

    k = st.slider("Number of clusters", 2, 10, 4)
    inertia = []
    for k_ in range(2, 11):
        km = KMeans(n_clusters=k_, random_state=42, n_init=10)
        km.fit(df_encoded)
        inertia.append(km.inertia_)
    st.subheader("Elbow Method for K Selection")
    st.line_chart(pd.DataFrame({"Clusters": list(range(2,11)), "Inertia": inertia}).set_index("Clusters"))

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(df_encoded)
    df_clusters = df_clust.copy()
    df_clusters["Cluster"] = cluster_labels

    st.subheader("Cluster Personas Table")
    persona_table = df_clusters.groupby("Cluster")[features].agg(lambda x: x.value_counts().idxmax())
    st.dataframe(persona_table)

    st.subheader("Clustered Data Download")
    out = df.copy()
    out["Cluster"] = cluster_labels
    st.download_button("Download Clustered Data", out.to_csv(index=False), file_name="clustered_data.csv")

# --- Tab 4: Association Rule Mining ---
elif tab == "Association Rule Mining":
    st.header("Association Rule Mining (Apriori)")

    # Pick two columns for multi-select association
    multiselect_cols = ["Frequent_Categories", "Shopping_Challenges", "Desired_Features"]
    col1 = st.selectbox("First column for rules", multiselect_cols)
    col2 = st.selectbox("Second column for rules", multiselect_cols, index=1)

    # Preprocess: convert comma separated to one-hot
    d1 = df[col1].str.get_dummies(sep=',')
    d2 = df[col2].str.get_dummies(sep=',')
    basket = pd.concat([d1, d2], axis=1)

    min_support = st.slider("Min Support", 0.01, 0.3, 0.05, 0.01)
    min_conf = st.slider("Min Confidence", 0.1, 0.9, 0.3, 0.05)
    min_lift = st.slider("Min Lift", 0.5, 3.0, 1.1, 0.1)

    freq_items = apriori(basket, min_support=min_support, use_colnames=True)
    rules = association_rules(freq_items, metric="confidence", min_threshold=min_conf)
    rules = rules[rules['lift'] >= min_lift]
    rules = rules.sort_values("confidence", ascending=False).head(10)

    st.subheader("Top 10 Association Rules")
    if not rules.empty:
        st.write(rules[["antecedents", "consequents", "support", "confidence", "lift"]])
        st.download_button("Download Rules", rules.to_csv(index=False), file_name="association_rules.csv")
    else:
        st.write("No rules found for selected parameters.")
