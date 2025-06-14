import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from st_aggrid import AgGrid, GridOptionsBuilder

# Page setup
st.set_page_config(page_title="Fake Job Post Detection", layout="wide", initial_sidebar_state="expanded")

# Load model and vectorizer
model = joblib.load("random_forest_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Custom heading
st.markdown(
    "<h1 style='font-size: 3rem; color: #4B8BBE;'>🕵️ Fake Job Post Detector</h1>",
    unsafe_allow_html=True
)
st.markdown("Upload a CSV file to detect potentially fraudulent job listings. Stay alert, stay safe!")

# File upload
uploaded_file = st.file_uploader("📁 Upload your job postings CSV", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded and processed successfully!")

        # Preprocessing
        df['combined_text'] = df['title'].fillna('') + " " + \
                              df['company_profile'].fillna('') + " " + \
                              df['description'].fillna('')

        # Vectorize and predict
        X = vectorizer.transform(df['combined_text'])
        df["Predicted_Label"] = model.predict(X)
        df["Fraud_Probability"] = model.predict_proba(X)[:, 1]

        # Tabs
        tab1, tab2, tab3 = st.tabs(["📄 Full Data", "🔍 Top 10 Suspicious", "📊 Visual Insights"])

        with tab1:
            st.markdown("### 📋 All Job Posts with Predictions")
            threshold = st.slider("🔽 Set fraud probability threshold", 0.0, 1.0, 0.5, 
                                  help="Only show jobs above this fraud probability")

            filtered_df = df[df["Fraud_Probability"] >= threshold]
            st.write(f"Showing {len(filtered_df)} posts with fraud probability ≥ {threshold}")

            gb = GridOptionsBuilder.from_dataframe(
                filtered_df[["title", "company_profile", "description", "Fraud_Probability", "Predicted_Label"]]
            )
            gb.configure_pagination()
            gb.configure_side_bar()
            grid_options = gb.build()

            AgGrid(filtered_df, gridOptions=grid_options, height=400, fit_columns_on_grid_load=True)

        with tab2:
            st.markdown("### 🧨 Top 10 Suspicious Job Posts")
            top10 = df.sort_values("Fraud_Probability", ascending=False).head(10)
            AgGrid(top10[["title", "company_profile", "description", "Fraud_Probability"]], 
                   height=400, fit_columns_on_grid_load=True)

        with tab3:
            st.markdown("### 📈 Prediction Summary")
            st.bar_chart(df["Predicted_Label"].value_counts(), use_container_width=True)
            st.markdown("### 📉 Fraud Probability Spread")
            st.line_chart(df["Fraud_Probability"].sort_values().reset_index(drop=True), use_container_width=True)

        # Download button
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Results as CSV", csv, "predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Error while processing the file:\n\n{e}")

else:
    st.info("📌 Please upload a CSV file to begin.")
