# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .high-risk {
        background-color: #c71f2d;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
    }
    .medium-risk {
        background-color: #FFFF00;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_data():
    """Load model and data"""
    try:
        model = joblib.load('models/churn_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        predictions_df = pd.read_csv('outputs/churn_predictions_with_ids.csv')
        high_risk_df = pd.read_csv('outputs/churn_high_risk_with_ids.csv')
        performance_metrics = joblib.load('models/performance_metrics.pkl')
        return model, scaler, feature_names, predictions_df, high_risk_df, performance_metrics
    except FileNotFoundError:
        st.error("❌ Model files not found. Please run train.py first.")
        return None, None, None, None, None, None

def predict_single_customer(model, scaler, feature_names, input_data):
    """Predict churn probability for a single customer"""
    try:
        # Preprocess input data
        input_df = pd.DataFrame([input_data])

        # Ensure all features are present
        for feature in feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0

        input_df = input_df[feature_names]

        # Scale features
        input_scaled = scaler.transform(input_df)

        # Predict
        probability = model.predict_proba(input_scaled)[0][1]

        return probability
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🚨 Customer Churn Prediction Dashboard</h1>', unsafe_allow_html=True)

    # Load model and data
    model, scaler, feature_names, predictions_df, high_risk_df, performance_metrics = load_data()

    if model is None:
        return

    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Dashboard Section",
        ["📊 Overview", "🔍 High-Risk Customers", "🎯 Single Prediction", "📈 Model Insights", "📁 Batch Prediction"]
    )

    # Overview Tab
    if app_mode == "📊 Overview":
        st.header("📊 Model Performance Overview")

        # Key metrics - Using your provided metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Accuracy", f"{performance_metrics['accuracy']:.1%}")

        with col2:
            st.metric("Recall", f"{performance_metrics['recall']:.1%}")

        with col3:
            st.metric("Precision", f"{performance_metrics['precision']:.1%}")

        with col4:
            st.metric("High-Risk Customers", f"{len(high_risk_df)}")

        # Visualizations
        st.subheader("📈 Key Visualizations")

        col1, col2 = st.columns(2)

        with col1:
            st.image('images/contract_churn.png', caption='Churn % by Contract Type')
            st.image('images/tenure_churn.png', caption='Churn % by Tenure Months')

        with col2:
            st.image('images/payment_churn.png', caption='Churn % by Payment Method')
            st.image('images/confusion_matrix.png', caption='Confusion Matrix')

        col3, col4 = st.columns(2)

        with col3:
            st.image('images/pr_curve.png', caption='Precision-Recall Curve')

        with col4:
            st.image('images/feature_importance.png', caption='Feature Importance')

    # High-Risk Customers Tab
    elif app_mode == "🔍 High-Risk Customers":
        st.header("🔍 High-Risk Customers - Action Required")

        # Risk level filter
        risk_level = st.slider("Risk Threshold", 0.5, 1.0, 0.7, 0.05)

        filtered_high_risk = high_risk_df[high_risk_df['PredictedProb'] >= risk_level]

        st.metric("Customers Above Threshold", len(filtered_high_risk))

        # Display high-risk customers
        st.subheader(f"🚨 {len(filtered_high_risk)} Customers Need Immediate Attention")

        for idx, row in filtered_high_risk.iterrows():
            risk_class = "high-risk" if row['PredictedProb'] >= 0.8 else "medium-risk"
            st.markdown(f"""
            <div class="{risk_class}">
                <strong>Customer ID:</strong> {row['CustomerID']} | 
                <strong>Churn Probability:</strong> {row['PredictedProb']:.1%} |
                <strong>Actual Status:</strong> {'Churned' if row['TrueLabel'] == 1 else 'Retained'}
            </div>
            """, unsafe_allow_html=True)

        # Download button
        csv = filtered_high_risk.to_csv(index=False)
        st.download_button(
            label="📥 Download High-Risk List",
            data=csv,
            file_name=f"high_risk_customers_threshold_{risk_level}.csv",
            mime="text/csv"
        )

        # Detailed table
        st.subheader("Detailed View")
        st.dataframe(filtered_high_risk.sort_values('PredictedProb', ascending=False))

    # Single Prediction Tab
    elif app_mode == "🎯 Single Prediction":
        st.header("🎯 Predict Churn for Single Customer")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Enter Customer Details")
            tenure_months = st.number_input("Tenure Months", min_value=0, max_value=100, value=24)
            monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=65.0)
            contract = st.selectbox("Contract Type", [0, 1, 2],
                                    format_func=lambda x: ["Month-to-month", "One year", "Two year"][x])
            online_security = st.selectbox("Online Security", [0, 1], format_func=lambda x: ["No", "Yes"][x])

        with col2:
            st.subheader("Additional Details")
            paperless_billing = st.selectbox("Paperless Billing", [0, 1], format_func=lambda x: ["No", "Yes"][x])
            payment_method = st.selectbox("Payment Method", [0, 1, 2, 3],
                                          format_func=lambda x:
                                          ["Electronic check", "Mailed check", "Bank transfer", "Credit card"][x])
            tech_support = st.selectbox("Tech Support", [0, 1], format_func=lambda x: ["No", "Yes"][x])
            cltv = st.number_input("CLTV", min_value=0, value=4000)

        if st.button("🔮 Predict Churn Probability"):
            # Create input data
            input_data = {
                'Tenure Months': tenure_months,
                'Monthly Charges': monthly_charges,
                'Contract': contract,
                'Online Security': online_security,
                'Paperless Billing': paperless_billing,
                'Payment Method': payment_method,
                'Tech Support': tech_support,
                'CLTV': cltv
            }

            # Add missing features with default values
            for feature in feature_names:
                if feature not in input_data:
                    input_data[feature] = 0

            probability = predict_single_customer(model, scaler, feature_names, input_data)

            if probability is not None:
                # Display result
                st.subheader("Prediction Result")

                if probability >= 0.7:
                    st.error(f"🚨 HIGH RISK - Churn Probability: {probability:.1%}")
                    st.warning("Immediate retention action recommended!")
                elif probability >= 0.5:
                    st.warning(f"⚠️ MEDIUM RISK - Churn Probability: {probability:.1%}")
                    st.info("Proactive engagement recommended")
                else:
                    st.success(f"✅ LOW RISK - Churn Probability: {probability:.1%}")
                    st.info("Customer appears stable")

                # Progress bar
                st.progress(probability)

    # Model Insights Tab
    elif app_mode == "📈 Model Insights":
        st.header("📈 Model Performance Insights")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Performance Metrics")

            # Using your provided metrics
            metrics_df = pd.DataFrame({
                'Metric': ['Accuracy', 'Recall', 'Precision', 'F1-Score'],
                'Value': [
                    performance_metrics['accuracy'],
                    performance_metrics['recall'],
                    performance_metrics['precision'],
                    performance_metrics['f1_score']
                ]
            })

            st.dataframe(metrics_df.style.format({'Value': '{:.1%}'}))

        with col2:
            st.subheader("Risk Distribution")
            risk_bins = pd.cut(predictions_df['PredictedProb'],
                               bins=[0, 0.3, 0.5, 0.7, 1.0],
                               labels=['Low', 'Medium', 'High', 'Critical'])
            risk_counts = risk_bins.value_counts()

            fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                         title="Customer Risk Distribution")
            st.plotly_chart(fig)

        # Feature importance interactive chart
        st.subheader("Feature Importance Analysis")
        feature_importance = model.feature_importances_
        feature_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=True)

        fig = px.bar(feature_df.tail(10), x='Importance', y='Feature',
                     orientation='h', title='Top 10 Most Important Features')
        st.plotly_chart(fig)

    # Batch Prediction Tab
    elif app_mode == "📁 Batch Prediction":
        st.header("📁 Batch Prediction - Upload CSV File")

        st.info("""
        📋 Upload a CSV file with customer data to get churn predictions for multiple customers.
        The file should contain the same features used during model training.
        """)

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            try:
                # Read uploaded file
                new_data = pd.read_csv(uploaded_file)

                st.success(f"✅ Successfully uploaded {len(new_data)} customers")

                # Check if CustomerID column exists
                if 'CustomerID' not in new_data.columns:
                    st.warning("⚠️ CustomerID column not found. Generating temporary IDs.")
                    new_data['CustomerID'] = [f"TEMP_{i}" for i in range(len(new_data))]

                # Display preview
                st.subheader("Data Preview")
                st.dataframe(new_data.head())

                if st.button("🚀 Predict Churn for All Customers"):
                    with st.spinner("Processing predictions..."):
                        # Prepare features
                        prediction_data = new_data.copy()

                        # Ensure all features are present
                        for feature in feature_names:
                            if feature not in prediction_data.columns:
                                prediction_data[feature] = 0

                        features_for_prediction = prediction_data[feature_names]

                        # Scale features
                        features_scaled = scaler.transform(features_for_prediction)

                        # Predict
                        probabilities = model.predict_proba(features_scaled)[:, 1]
                        predictions = (probabilities >= 0.5).astype(int)

                        # Create results
                        results = pd.DataFrame({
                            'CustomerID': prediction_data['CustomerID'],
                            'ChurnProbability': probabilities,
                            'PredictedChurn': predictions,
                            'RiskLevel': pd.cut(probabilities,
                                                bins=[0, 0.3, 0.5, 0.7, 1.0],
                                                labels=['Low', 'Medium', 'High', 'Critical'])
                        })

                        st.subheader("📊 Prediction Results")

                        # Summary metrics
                        col1, col2, col3 = st.columns(3)
                        high_risk_count = (results['ChurnProbability'] >= 0.7).sum()

                        with col1:
                            st.metric("Total Customers", len(results))
                        with col2:
                            st.metric("High-Risk Customers", high_risk_count)
                        with col3:
                            st.metric("High-Risk Percentage", f"{(high_risk_count / len(results)):.1%}")

                        # Display results
                        st.dataframe(results.sort_values('ChurnProbability', ascending=False))

                        # Download results
                        csv = results.to_csv(index=False)
                        st.download_button(
                            label="📥 Download All Predictions",
                            data=csv,
                            file_name="batch_churn_predictions.csv",
                            mime="text/csv"
                        )

            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown(
        "### 💡 About This Dashboard\n"
        "This dashboard provides real-time churn prediction insights using a trained Random Forest model. "
        "Use it to identify at-risk customers and take proactive retention actions."
    )

if __name__ == "__main__":
    main()