# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and preprocessors
model = joblib.load("model/best_random_forest.pkl")
scaler = joblib.load("model/scaler.pkl")
model_columns = joblib.load("model/model_columns.pkl")

st.set_page_config(page_title="🎓 Lead Conversion Predictor", layout="centered")
st.title("🎓 Educational Lead Conversion Predictor")
st.markdown("Predict which leads are likely to convert using metadata and AI.")

# --- PREDICT FUNCTION ---
def preprocess_and_predict(input_df):
    df = input_df.copy()
    df = df.reindex(columns=model_columns, fill_value=0)
    df_scaled = scaler.transform(df)
    preds = model.predict(df_scaled)
    return preds


# --- SINGLE INPUT FORM (Simplified) ---
with st.expander("📋 Predict One Lead (Simple)"):
    st.markdown("Enter key info for the lead:")

    total_time = st.number_input("🕒 Total Time Spent on Website", min_value=0, help="Time (in seconds) the user spent on your website")
    total_visits = st.number_input("🔁 Total Visits", min_value=0, help="How many times this lead has visited")
    page_views = st.number_input("📄 Page Views Per Visit", min_value=0, help="Average number of pages viewed per visit")
    
    tags = st.selectbox("🏷️ Tags", ['Interested in course', 'Ringing', 'Will revert after reading email', 'Lost to EINS', 'Other'])
    lead_source = st.selectbox("🌐 Lead Source", ['Google', 'Direct Traffic', 'Reference', 'Social Media', 'Other'])
    last_activity = st.selectbox("⚡ Last Activity", ['Email Opened', 'Page Visited on Website', 'Form Submitted', 'Olark Chat Conversation', 'Other'])
    specialization = st.selectbox("🎓 Specialization", ['Finance', 'IT', 'Marketing', 'HR', 'Operations', 'Other'])

    # Create dummy input
    simple_input = pd.DataFrame(columns=model_columns)
    simple_input.loc[0] = [0]*len(model_columns)# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="🎓 Lead Conversion Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .success-prediction {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
    }
    .failed-prediction {
        background: linear-gradient(90deg, #ff416c 0%, #ff4b2b 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
</style>
""", unsafe_allow_html=True)

# Load model and preprocessors with error handling
@st.cache_resource
def load_models():
    try:
        model = joblib.load("model/best_random_forest.pkl")
        scaler = joblib.load("model/scaler.pkl")
        model_columns = joblib.load("model/model_columns.pkl")
        return model, scaler, model_columns
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

model, scaler, model_columns = load_models()

# Header
st.markdown("""
<div class="main-header">
    <h1>🎓 Educational Lead Conversion Predictor</h1>
    <p>Advanced ML-powered predictions with comprehensive analytics and insights</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("🎯 Navigation")
page = st.sidebar.selectbox("Choose a page:", [
    "🏠 Dashboard",
    "🔮 Single Prediction",
    "📊 Batch Analysis",
    "📈 Model Insights",
    "📋 Data Explorer"
])

# --- HELPER FUNCTIONS ---
def preprocess_and_predict(input_df):
    """Preprocess input data and make predictions"""
    df = input_df.copy()
    df = df.reindex(columns=model_columns, fill_value=0)
    df_scaled = scaler.transform(df)
    preds = model.predict(df_scaled)
    probabilities = model.predict_proba(df_scaled)
    return preds, probabilities

def get_feature_importance():
    """Get feature importance from the model"""
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': model_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        return importance_df
    return None

def create_prediction_gauge(probability):
    """Create a gauge chart for prediction probability"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Conversion Probability (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 25], 'color': "lightgray"},
                {'range': [25, 50], 'color': "gray"},
                {'range': [50, 75], 'color': "lightgreen"},
                {'range': [75, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

# --- DASHBOARD PAGE ---
if page == "🏠 Dashboard":
    st.header("📊 Dashboard Overview")
    
    # Key metrics (mock data - replace with actual data)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📈 Total Predictions</h3>
            <h2>1,247</h2>
            <p>+15% from last month</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>✅ Conversion Rate</h3>
            <h2>68.2%</h2>
            <p>+3.2% from last month</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Model Accuracy</h3>
            <h2>92.5%</h2>
            <p>Validated on test data</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡ Avg Response Time</h3>
            <h2>0.12s</h2>
            <p>Real-time predictions</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sample visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Conversion Trends")
        # Mock data for trend chart
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='M')
        conversion_rates = np.random.uniform(0.6, 0.8, len(dates))
        
        fig = px.line(
            x=dates, 
            y=conversion_rates,
            title="Monthly Conversion Rate Trend",
            labels={'x': 'Date', 'y': 'Conversion Rate'}
        )
        fig.update_traces(line=dict(color='#667eea', width=3))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Lead Sources Distribution")
        # Mock data for pie chart
        sources = ['Google', 'Direct Traffic', 'Reference', 'Social Media', 'Other']
        values = [35, 25, 20, 15, 5]
        
        fig = px.pie(
            values=values,
            names=sources,
            title="Lead Source Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)

# --- SINGLE PREDICTION PAGE ---
elif page == "🔮 Single Prediction":
    st.header("🔮 Single Lead Prediction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Lead Information")
        
        # Input form with better organization
        with st.form("prediction_form"):
            col_a, col_b = st.columns(2)
            
            with col_a:
                total_time = st.number_input(
                    "🕒 Total Time Spent (seconds)",
                    min_value=0,
                    value=300,
                    help="Total time spent on website in seconds"
                )
                total_visits = st.number_input(
                    "🔁 Total Visits",
                    min_value=0,
                    value=3,
                    help="Number of times the lead visited"
                )
                page_views = st.number_input(
                    "📄 Page Views Per Visit",
                    min_value=0,
                    value=5,
                    help="Average pages viewed per visit"
                )
            
            with col_b:
                tags = st.selectbox(
                    "🏷️ Tags",
                    ['Interested in course', 'Ringing', 'Will revert after reading email', 'Lost to EINS', 'Other']
                )
                lead_source = st.selectbox(
                    "🌐 Lead Source",
                    ['Google', 'Direct Traffic', 'Reference', 'Social Media', 'Other']
                )
                last_activity = st.selectbox(
                    "⚡ Last Activity",
                    ['Email Opened', 'Page Visited on Website', 'Form Submitted', 'Olark Chat Conversation', 'Other']
                )
                specialization = st.selectbox(
                    "🎓 Specialization",
                    ['Finance', 'IT', 'Marketing', 'HR', 'Operations', 'Other']
                )
            
            submitted = st.form_submit_button("🎯 Predict Conversion", type="primary")
        
        if submitted:
            # Create input dataframe
            simple_input = pd.DataFrame(columns=model_columns)
            simple_input.loc[0] = [0] * len(model_columns)
            
            # Set values
            simple_input.at[0, 'Total Time Spent on Website'] = total_time
            simple_input.at[0, 'TotalVisits'] = total_visits
            simple_input.at[0, 'Page Views Per Visit'] = page_views
            simple_input.at[0, f'Tags_{tags}'] = 1
            simple_input.at[0, f'Lead Source_{lead_source}'] = 1
            simple_input.at[0, f'Last Activity_{last_activity}'] = 1
            simple_input.at[0, f'Specialization_{specialization}'] = 1
            
            # Make prediction
            pred, prob = preprocess_and_predict(simple_input)
            conversion_prob = prob[0][1]  # Probability of conversion
            
            # Store results in session state for visualization
            st.session_state.prediction = pred[0]
            st.session_state.probability = conversion_prob
    
    with col2:
        st.subheader("🎯 Prediction Results")
        
        if hasattr(st.session_state, 'prediction'):
            # Display prediction
            if st.session_state.prediction == 1:
                st.markdown("""
                <div class="success-prediction">
                    <h3>✅ LIKELY TO CONVERT</h3>
                    <p>This lead shows high conversion potential!</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="failed-prediction">
                    <h3>❌ UNLIKELY TO CONVERT</h3>
                    <p>This lead may need more nurturing.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Display probability gauge
            fig = create_prediction_gauge(st.session_state.probability)
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.subheader("💡 Recommendations")
            if st.session_state.probability > 0.7:
                st.success("🎯 **Priority Lead**: Contact immediately!")
                st.info("📞 Recommended actions: Direct call, personalized email")
            elif st.session_state.probability > 0.4:
                st.warning("⚠️ **Moderate Interest**: Nurture with content")
                st.info("📧 Recommended actions: Email campaign, webinar invitation")
            else:
                st.error("📉 **Low Interest**: Long-term nurturing needed")
                st.info("📚 Recommended actions: Educational content, newsletter")

# --- BATCH ANALYSIS PAGE ---
elif page == "📊 Batch Analysis":
    st.header("📊 Batch Lead Analysis")
    
    uploaded_file = st.file_uploader(
        "📁 Upload CSV file with lead data",
        type=["csv"],
        help="Upload a CSV file containing lead information for batch prediction"
    )
    
    if uploaded_file:
        try:
            input_csv = pd.read_csv(uploaded_file)
            
            # Display file info
            st.subheader("📋 File Information")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Total Rows", len(input_csv))
            with col2:
                st.metric("📈 Columns", len(input_csv.columns))
            with col3:
                st.metric("💾 File Size", f"{uploaded_file.size / 1024:.1f} KB")
            
            # Preview data
            st.subheader("👀 Data Preview")
            st.dataframe(input_csv.head(10), use_container_width=True)
            
            if st.button("🔮 Generate Predictions", type="primary"):
                with st.spinner("Processing predictions..."):
                    predictions, probabilities = preprocess_and_predict(input_csv)
                    
                    # Add results to dataframe
                    input_csv['Prediction'] = predictions
                    input_csv['Conversion_Probability'] = probabilities[:, 1]
                    input_csv['Risk_Level'] = pd.cut(
                        probabilities[:, 1],
                        bins=[0, 0.3, 0.7, 1.0],
                        labels=['Low', 'Medium', 'High']
                    )
                
                # Display results
                st.subheader("📊 Analysis Results")
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("✅ Predicted Conversions", sum(predictions))
                with col2:
                    st.metric("📈 Conversion Rate", f"{(sum(predictions)/len(predictions)*100):.1f}%")
                with col3:
                    st.metric("🎯 High Risk Leads", sum(input_csv['Risk_Level'] == 'High'))
                with col4:
                    st.metric("📊 Average Probability", f"{probabilities[:, 1].mean():.3f}")
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Probability distribution
                    fig = px.histogram(
                        input_csv,
                        x='Conversion_Probability',
                        nbins=20,
                        title="Conversion Probability Distribution",
                        labels={'Conversion_Probability': 'Probability', 'count': 'Number of Leads'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Risk level distribution
                    risk_counts = input_csv['Risk_Level'].value_counts()
                    fig = px.pie(
                        values=risk_counts.values,
                        names=risk_counts.index,
                        title="Lead Risk Distribution",
                        color_discrete_map={'Low': '#ff4444', 'Medium': '#ffaa00', 'High': '#44ff44'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed results
                st.subheader("📋 Detailed Results")
                st.dataframe(input_csv, use_container_width=True)
                
                # Download results
                csv_download = input_csv.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Results",
                    csv_download,
                    f"lead_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    key='download-csv'
                )
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure your CSV file has the correct format and column names.")

# --- MODEL INSIGHTS PAGE ---
elif page == "📈 Model Insights":
    st.header("📈 Model Performance & Insights")
    
    # Feature importance
    importance_df = get_feature_importance()
    
    if importance_df is not None:
        st.subheader("🔍 Feature Importance Analysis")
        
        # Top 10 most important features
        top_features = importance_df.head(10)
        
        fig = px.bar(
            top_features,
            x='importance',
            y='feature',
            orientation='h',
            title="Top 10 Most Important Features",
            labels={'importance': 'Importance Score', 'feature': 'Features'}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance table
        st.subheader("📊 All Features Importance")
        st.dataframe(importance_df, use_container_width=True)
    
    # Model performance metrics (mock data)
    st.subheader("📊 Model Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Confusion matrix (mock data)
        conf_matrix = np.array([[180, 20], [15, 185]])
        fig = px.imshow(
            conf_matrix,
            text_auto=True,
            aspect="auto",
            title="Confusion Matrix",
            labels=dict(x="Predicted", y="Actual", color="Count")
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # ROC curve (mock data)
        fpr = np.linspace(0, 1, 100)
        tpr = np.sqrt(fpr)  # Mock ROC curve
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
        fig.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate"
        )
        st.plotly_chart(fig, use_container_width=True)

# --- DATA EXPLORER PAGE ---
elif page == "📋 Data Explorer":
    st.header("📋 Data Explorer & Analytics")
    
    st.info("Upload a dataset to explore data patterns and relationships")
    
    uploaded_file = st.file_uploader("📁 Upload CSV for exploration", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        st.subheader("📊 Dataset Overview")
        
        # Basic info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Rows", len(df))
        with col2:
            st.metric("📈 Columns", len(df.columns))
        with col3:
            st.metric("💾 Memory Usage", f"{df.memory_usage().sum() / 1024:.1f} KB")
        
        # Data preview
        st.subheader("👀 Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        
        # Statistical summary
        st.subheader("📈 Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
        
        # Column analysis
        st.subheader("🔍 Column Analysis")
        selected_column = st.selectbox("Select column to analyze:", df.columns)
        
        if df[selected_column].dtype in ['int64', 'float64']:
            # Numeric column
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(df, x=selected_column, title=f"Distribution of {selected_column}")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.box(df, y=selected_column, title=f"Box Plot of {selected_column}")
                st.plotly_chart(fig, use_container_width=True)
        else:
            # Categorical column
            value_counts = df[selected_column].value_counts()
            fig = px.bar(x=value_counts.index, y=value_counts.values, 
                        title=f"Distribution of {selected_column}")
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🎓 Educational Lead Conversion Predictor v2.0 | Built with ❤️ using Streamlit</p>
    <p>📊 Powered by Machine Learning | 🚀 Real-time Predictions</p>
</div>
""", unsafe_allow_html=True)

    # Set only selected values
simple_input.at[0, 'Total Time Spent on Website'] = total_time
simple_input.at[0, 'TotalVisits'] = total_visits
simple_input.at[0, 'Page Views Per Visit'] = page_views
simple_input.at[0, f'Tags_{tags}'] = 1
simple_input.at[0, f'Lead Source_{lead_source}'] = 1
simple_input.at[0, f'Last Activity_{last_activity}'] = 1
simple_input.at[0, f'Specialization_{specialization}'] = 1

if st.button("🎯 Predict Now"):
        pred = preprocess_and_predict(simple_input)[0]
        st.success(f"🎯 Prediction: {'✅ Converted' if pred == 1 else '❌ Not Converted'}")



# --- BATCH CSV UPLOAD ---
st.markdown("---")
st.header("📁 Batch Prediction from CSV")
uploaded_file = st.file_uploader("Upload a leads CSV file", type=["csv"])

if uploaded_file:
    input_csv = pd.read_csv(uploaded_file)
    st.write("📝 Input Preview", input_csv.head())
    try:
        predictions = preprocess_and_predict(input_csv)
        input_csv['Prediction'] = predictions
        st.write("✅ Prediction Results", input_csv)

        # Download result
        csv = input_csv.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results", csv, "lead_predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"Error in prediction: {e}")
