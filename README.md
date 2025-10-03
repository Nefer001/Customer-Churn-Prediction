# 🚨 Customer Churn Prediction App

An AI-powered churn prediction system built with Random Forests, Streamlit, and visual analytics.
The project helps businesses identify at-risk customers, understand key churn drivers, and take proactive retention actions.

### Features
**🔹 Machine Learning Pipeline (train.py)**
  - Cleans and preprocesses Telco customer churn dataset
  - Encodes categorical variables, handles missing values
  - Performs oversampling to balance churn vs. non-churn classes
  - Trains a Random Forest Classifier with feature scaling
  - Saves trained model, scaler, and feature metadata (.pkl)
  - Generates key visualizations:
    - Churn % by Contract
    - Churn % by Tenure Group
    - Churn % by Payment Method
    - Confusion Matrix (threshold = 0.3)
    - Precision–Recall Curve
    - Top 10 Feature Importance
    
#### **🔹 Interactive Dashboard (streamlit_app.py)**
  - **Overview**: Performance metrics & visual insights
  - **High-Risk Customers**: Filter customers by churn probability threshold, export lists
  - **Single Prediction**: Predict churn probability for a new customer profile
  - **Model Insights**: Risk distribution & feature importance analysis
  - **Batch Prediction**: Upload CSV file for churn predictions at scale
  - **Downloadable Reports**: Export predictions to CSV

### **Project Structure** 
customer-churn-prediction/
├── src/
│ ├── train.py # Model training pipeline
│ └── streamlit_app.py # Interactive dashboard
├── data/
│ └── Telco_customer_churn.csv
├── models/ # Serialized models & scalers
├── outputs/ # Prediction results
├── images/ # Visualizations & charts
├── requirements.txt
└── README.md

###  Tech Stack
  - **Python**: pandas, numpy, scikit-learn, joblib
  - **Visualization**: matplotlib, seaborn, plotly
  - **Model**: Random Forest Classifier
  - **App**: Streamlit

###  Getting Started
#### 1️⃣ **Clone the Repository**
  - git clone https://github.com/Nefer001/Customer-Churn-Prediction.git
  - cd churn-dashboard

#### 2️⃣ **Install Dependencies**
  - pip install -r requirements.txt

#### 3️⃣ Train the Model
cd src
`python train.py`

This generates:
  - Trained model & scaler (/models/)
  - Predictions (/outputs/)
  - Visualizations (.png)

#### 4️⃣ Launch the Dashboard
`streamlit run streamlit_app.py`

Open http://localhost:8501 in your browser.

###  Use Cases
  - ✅ Identify high-risk customers before they churn
  - ✅ Target retention campaigns more effectively
  - ✅ Understand which features drive churn the most
  - ✅ Batch process entire customer lists

###  Demo
👉 After running the app, you’ll see tabs for:

 - **Overview**: Performance & plots
 - **High-Risk Customers**: Actionable churn list
 - **Single Prediction**: Predict for new customer details
 - **Model Insights**: Deeper look into feature importance
 - **Batch Prediction**: Upload CSV for mass prediction

### Author

Developed by [Edward Deodatus]

AI & Data Science Portfolio Project 

If you found this helpful, ⭐ the repo and connect with me!