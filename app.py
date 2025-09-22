import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="Stroke Prediction App",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .high-risk {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .high-risk h3 {
        color: #c62828;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .high-risk p {
        color: #b71c1c;
        font-weight: 600;
    }
    .high-risk strong {
        color: #8b0000;
        background-color: #ffcdd2;
        padding: 2px 6px;
        border-radius: 3px;
        font-weight: bold;
    }
    .low-risk {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
    }
    .low-risk h3 {
        color: #2e7d32;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .low-risk p {
        color: #1b5e20;
        font-weight: 600;
    }
    .low-risk strong {
        color: #0d5016;
        background-color: #c8e6c9;
        padding: 2px 6px;
        border-radius: 3px;
        font-weight: bold;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box h3 {
        color: #1565c0;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .info-box p {
        color: #263238;
        font-weight: 500;
        line-height: 1.6;
        margin-bottom: 0.8rem;
    }
    .info-box strong {
        color: #d32f2f;
        background-color: #ffebee;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: bold;
        border-left: 3px solid #f44336;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model with error handling"""
    model_path = 'best_model.pkl'
    
    if not os.path.exists(model_path):
        # Create a dummy model for demonstration if the actual model doesn't exist
        st.warning("Model file not found. Using a demo model for illustration purposes.")
        
        # Create a simple demo model
        from sklearn.ensemble import RandomForestClassifier
        demo_model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Create dummy training data
        X_dummy = np.random.rand(100, 10)
        y_dummy = np.random.randint(0, 2, 100)
        demo_model.fit(X_dummy, y_dummy)
        
        return demo_model, True  # True indicates it's a demo model
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model, False  # False indicates it's the real model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

def create_input_dataframe(gender, age, hypertension, heart_disease, ever_married, 
                          work_type, residence_type, avg_glucose_level, bmi, smoking_status):
    """Create a properly formatted DataFrame from user inputs"""
    
    # Define mappings
    gender_map = {'Female': 0, 'Male': 1, 'Other': 2}
    ever_married_map = {'No': 0, 'Yes': 1}
    work_type_map = {'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'children': 3, 'Never_worked': 4}
    residence_type_map = {'Rural': 0, 'Urban': 1}
    smoking_status_map = {'Unknown': 0, 'formerly smoked': 1, 'never smoked': 2, 'smokes': 3}
    
    # Create DataFrame
    input_data = pd.DataFrame([[
        gender_map[gender],
        age,
        hypertension,
        heart_disease,
        ever_married_map[ever_married],
        work_type_map[work_type],
        residence_type_map[residence_type],
        avg_glucose_level,
        bmi,
        smoking_status_map[smoking_status]
    ]], columns=[
        'gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 
        'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'
    ])
    
    return input_data

def create_risk_gauge(risk_score):
    """Create a risk gauge visualization"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Stroke Risk %"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 25], 'color': "lightgreen"},
                {'range': [25, 50], 'color': "yellow"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "red"}
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

def main():
    # Title and introduction
    st.markdown('<h1 class="main-header">🧠 Stroke Prediction App</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h3>About This App</h3>
    <p>This application uses machine learning to predict the likelihood of stroke based on various health and demographic factors. 
    Please enter the patient's information in the sidebar to get a prediction.</p>
    <p><strong>Disclaimer:</strong> This tool is for educational purposes only and should not replace professional medical advice.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model, is_demo = load_model()
    
    if is_demo:
        st.info("📝 **Demo Mode**: Using a demonstration model. For production use, please ensure 'best_model.pkl' is available.")
    
    # Sidebar for input
    st.sidebar.header("🔍 Patient Information")
    st.sidebar.markdown("---")
    
    # Demographic information
    st.sidebar.subheader("👤 Demographics")
    gender = st.sidebar.selectbox(
        "Gender", 
        options=['Female', 'Male', 'Other'],
        help="Patient's gender"
    )
    
    age = st.sidebar.slider(
        "Age", 
        min_value=0.0, 
        max_value=100.0, 
        value=45.0, 
        step=0.1,
        help="Patient's age in years"
    )
    
    ever_married = st.sidebar.radio(
        "Marital Status", 
        options=['No', 'Yes'],
        help="Has the patient ever been married?"
    )
    
    # Medical history
    st.sidebar.subheader("🏥 Medical History")
    hypertension = st.sidebar.radio(
        "Hypertension", 
        options=[0, 1], 
        format_func=lambda x: "Yes" if x == 1 else "No",
        help="Does the patient have hypertension?"
    )
    
    heart_disease = st.sidebar.radio(
        "Heart Disease", 
        options=[0, 1], 
        format_func=lambda x: "Yes" if x == 1 else "No",
        help="Does the patient have heart disease?"
    )
    
    # Lifestyle factors
    st.sidebar.subheader("🏃 Lifestyle")
    work_type = st.sidebar.selectbox(
        "Work Type", 
        options=['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'],
        help="Patient's work type"
    )
    
    residence_type = st.sidebar.radio(
        "Residence Type", 
        options=['Rural', 'Urban'],
        help="Type of residence"
    )
    
    smoking_status = st.sidebar.selectbox(
        "Smoking Status", 
        options=['never smoked', 'formerly smoked', 'smokes', 'Unknown'],
        help="Patient's smoking history"
    )
    
    # Health metrics
    st.sidebar.subheader("📊 Health Metrics")
    avg_glucose_level = st.sidebar.number_input(
        "Average Glucose Level (mg/dL)", 
        min_value=50.0, 
        max_value=300.0, 
        value=106.0, 
        step=0.1,
        help="Average glucose level in mg/dL"
    )
    
    bmi = st.sidebar.number_input(
        "BMI (Body Mass Index)", 
        min_value=10.0, 
        max_value=60.0, 
        value=28.1, 
        step=0.1,
        help="Body Mass Index"
    )
    
    # Add BMI interpretation
    if bmi < 18.5:
        bmi_category = "Underweight"
        bmi_color = "blue"
    elif bmi < 25:
        bmi_category = "Normal weight"
        bmi_color = "green"
    elif bmi < 30:
        bmi_category = "Overweight"
        bmi_color = "orange"
    else:
        bmi_category = "Obese"
        bmi_color = "red"
    
    st.sidebar.markdown(f"BMI Category: <span style='color: {bmi_color}'><strong>{bmi_category}</strong></span>", unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # Prediction button
    predict_button = st.sidebar.button(
        "🔮 Predict Stroke Risk", 
        type="primary",
        use_container_width=True
    )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    if predict_button:
        # Create input DataFrame
        input_data = create_input_dataframe(
            gender, age, hypertension, heart_disease, ever_married,
            work_type, residence_type, avg_glucose_level, bmi, smoking_status
        )
        
        # Make prediction
        try:
            if hasattr(model, 'predict_proba'):
                # If model has predict_proba (classification)
                prediction_proba = model.predict_proba(input_data)
                risk_score = prediction_proba[0][1] if len(prediction_proba[0]) > 1 else prediction_proba[0][0]
            else:
                # If model only has predict (regression)
                prediction = model.predict(input_data)
                risk_score = max(0, min(1, prediction[0]))  # Ensure it's between 0 and 1
            
            with col1:
                st.subheader("🎯 Prediction Results")
                
                # Risk assessment
                if risk_score > 0.5:
                    st.markdown(f"""
                    <div class="prediction-box high-risk">
                        <h3>⚠️ High Risk</h3>
                        <p><strong>Risk Score: {risk_score:.1%}</strong></p>
                        <p>The model indicates a higher likelihood of stroke risk. Please consult with a healthcare professional for proper evaluation and preventive measures.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-box low-risk">
                        <h3>✅ Lower Risk</h3>
                        <p><strong>Risk Score: {risk_score:.1%}</strong></p>
                        <p>The model indicates a lower likelihood of stroke risk. Continue maintaining healthy lifestyle habits.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show input data
                st.subheader("📋 Input Summary")
                
                # Create a more readable summary
                summary_data = {
                    "Parameter": ["Gender", "Age", "Hypertension", "Heart Disease", "Ever Married", 
                                "Work Type", "Residence", "Avg Glucose", "BMI", "Smoking Status"],
                    "Value": [gender, f"{age} years", "Yes" if hypertension else "No", 
                            "Yes" if heart_disease else "No", ever_married, work_type, 
                            residence_type, f"{avg_glucose_level} mg/dL", f"{bmi} ({bmi_category})", smoking_status]
                }
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.subheader("📊 Risk Visualization")
                
                # Risk gauge
                gauge_fig = create_risk_gauge(risk_score)
                st.plotly_chart(gauge_fig, use_container_width=True)
                
                # Risk factors breakdown
                st.subheader("🔍 Key Risk Factors")
                
                risk_factors = []
                if hypertension:
                    risk_factors.append("Hypertension")
                if heart_disease:
                    risk_factors.append("Heart Disease")
                if age > 65:
                    risk_factors.append("Advanced Age")
                if bmi > 30:
                    risk_factors.append("Obesity")
                if avg_glucose_level > 125:
                    risk_factors.append("High Glucose")
                if smoking_status == "smokes":
                    risk_factors.append("Current Smoking")
                
                if risk_factors:
                    for factor in risk_factors:
                        st.markdown(f"• {factor}")
                else:
                    st.markdown("• No major risk factors identified")
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.info("Please check your input data and try again.")
    
    else:
        with col1:
            st.info("👈 Please enter patient information in the sidebar and click 'Predict Stroke Risk' to get started.")
            
            # Show some general information about stroke risk factors
            st.subheader("🧠 About Stroke Risk Factors")
            
            st.markdown("""
            **Major Risk Factors for Stroke:**
            
            - **Age**: Risk increases with age, especially after 55
            - **High Blood Pressure**: The most important controllable risk factor
            - **Heart Disease**: Various heart conditions increase stroke risk
            - **Diabetes**: High blood sugar damages blood vessels
            - **High Cholesterol**: Can lead to blocked arteries
            - **Smoking**: Damages blood vessels and increases clot formation
            - **Obesity**: Often associated with other risk factors
            
            **Lifestyle Modifications:**
            - Regular exercise
            - Healthy diet (low sodium, rich in fruits and vegetables)
            - Maintain healthy weight
            - Don't smoke
            - Limit alcohol consumption
            - Manage stress
            """)
        
        with col2:
            st.markdown("""
            ### 📞 Emergency Signs of Stroke
            
            **Remember F.A.S.T.:**
            - **F**ace drooping
            - **A**rm weakness
            - **S**peech difficulty
            - **T**ime to call emergency services
            
            ### 🩺 When to Consult a Doctor
            - Regular check-ups
            - If you have multiple risk factors
            - Family history of stroke
            - Sudden severe headaches
            - Vision problems
            """)

if __name__ == "__main__":
    main()