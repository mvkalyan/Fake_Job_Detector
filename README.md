🕵️ #Spot The Scam: Fake Job Post Detector
 
 
 
🚀 Overview
 
Spot The Scam is a Streamlit web app that uses machine learning to detect potentially fraudulent job postings from uploaded CSV files. It leverages a pre-trained Random Forest model and TF-IDF vectorization to flag suspicious posts based on title, description, and company profile.
 
 
---
 
🎯 Features
 
✅ Upload CSV of job posts
 
📊 Visualize fraud probability distribution
 
🔍 View top 10 most suspicious posts
 
🧠 ML model predicts probability of fraud
 
📥 Download predictions
 
🌗 Light/Dark mode toggle
 
📋 Enhanced table views with AgGrid
 
 
 
---
 
🧠 Model
 
Model: Random Forest Classifier
 
Vectorizer: TF-IDF (scikit-learn)
 
Training Accuracy: ~97.8%
 
Recall (Fraud class): ~52% with threshold tuning
 
 
 
---
 
🛠 How to Run
 
# 1. Clone the repository
$ git clone https://github.com/yourusername/spot-the-scam-app.git
$ cd spot-the-scam-app
 
# 2. Install requirements
$ pip install -r requirements.txt
 
# 3. Run Streamlit app
$ streamlit run streamlit_app.py
 
 
---
 
📂 Folder Structure
 
spot-the-scam-app/
├── streamlit_app.py              # Main Streamlit app
├── random_forest_model.pkl       # Trained model
├── tfidf_vectorizer.pkl          # Saved TF-IDF vectorizer
├── Spot_The_Scam.ipynb           # Jupyter notebook (optional)
├── requirements.txt              # Python dependencies
└── README.md                     # You're here
 
 
---
 
🔗 Demo Links
 
🔴 Demo Video: Watch on YouTube
 
🌐 Deployed App (Optional): Visit App
 
 
 
---
 
📄 Notes
 
This app does not retrain the model during runtime
 
Make sure to upload a clean CSV with the required columns: title, company_profile, description
 
 
 
---
