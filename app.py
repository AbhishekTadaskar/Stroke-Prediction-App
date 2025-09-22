import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Set page configuration
st.set_page_config(
    page_title="Stroke Prediction App", 
    page_icon="🧠",
    layout="wide"
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
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .high-risk {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .high-risk h3 {
        color: #c62828;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .high-risk p {
        color: #b71c1c;
        font-weight: 600;
        font-size: 1.1rem;
    }
    .high-risk strong {
        color: #8b0000;
        background-color: #ffcdd2;
        padding: 4px 8px;
        border-radius: 5px;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .low-risk {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
    }
    .low-risk h3 {
        color: #2e7d32;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .low-risk p {
        color: #1b5e20;
        font-weight: 600;
        font-size: 1.1rem;
    }
    .low-risk strong {
        color: #0d5016;
        background-color: #c8e6c9;
        padding: 4px 8px;
        border-radius: 5px;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #1565c0;
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
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
        border-left: 3px solid #f44336;
    }
    .dataframe-container {
        background-color: #fafafa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# --- Load the Model ---
@st.cache_resource
def load_model():
    model_file_path = "best_model.pkl"
    if not os.path.exists(model_file_path):
        st.error("Error: The 'best_model.pkl' file was not found. Please make sure the file is in the same directory.")
        st.stop()

    try:
        with open(model_file_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"An error occurred while loading the model file: {e}")
        st.stop()

# --- Helper Functions ---
def encode_input(input_df):
    """Encode categorical features to match training data"""
    df_encoded = input_df.copy()
    
    # Manually map categorical values to their encoded integer representations
    df_encoded['gender'] = df_encoded['gender'].map({'Male': 1, 'Female': 0, 'Other': 2})
    df_encoded['ever_married'] = df_encoded['ever_married'].map({'No': 0, 'Yes': 1})
    df_encoded['work_type'] = df_encoded['work_type'].map({
        'Govt_job': 0, 
        'Never_worked': 1, 
        'Private': 2, 
        'Self-employed': 3, 
        'children': 4
    })
    df_encoded['Residence_type'] = df_encoded['Residence_type'].map({'Rural': 0, 'Urban': 1})
    df_encoded['smoking_status'] = df_encoded['smoking_status'].map({
        'Unknown': 0, 
        'formerly smoked': 1, 
        'never smoked': 2, 
        'smokes': 3
    })
    
    return df_encoded

# --- App Title and Description ---
st.markdown('<h1 class="main-header">🧠 Stroke Prediction App</h1>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
<h3>About This App</h3>
<p>This application uses machine learning to predict the likelihood of stroke based on various health and demographic factors. 
Please enter the patient's information in the sidebar to get a prediction.</p>
<p><strong>Disclaimer:</strong> This tool is for educational purposes only and should not replace professional medical advice.</p>
</div>
""", unsafe_allow_html=True)

# Load the model
model = load_model()
st.success("✅ Model loaded successfully!")

# --- User Input Fields ---
st.sidebar.header("🔍 Patient Data Input")
st.sidebar.markdown("---")

with st.sidebar.form("input_form"):
    st.subheader("👤 Demographics")
    gender = st.selectbox("Gender", options=['Male', 'Female', 'Other'])
    age = st.slider("Age", 0, 120, 45)
    ever_married = st.selectbox("Ever Married", options=['No', 'Yes'])
    
    st.subheader("🏥 Medical History") 
    hypertension = st.selectbox("Hypertension", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    heart_disease = st.selectbox("Heart Disease", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    st.subheader("🏃 Lifestyle")
    work_type = st.selectbox("Work Type", options=['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'])
    residence_type = st.selectbox("Residence Type", options=['Urban', 'Rural'])
    smoking_status = st.selectbox("Smoking Status", options=['formerly smoked', 'never smoked', 'smokes', 'Unknown'])
    
    st.subheader("📊 Health Metrics")
    avg_glucose_level = st.slider("Average Glucose Level (mg/dL)", 50.0, 300.0, 100.0)
    bmi = st.slider("BMI", 10.0, 60.0, 25.0)

    submit_button = st.form_submit_button("🔮 Predict Stroke Risk", type="primary")

# --- Main Content Area ---
col1, col2 = st.columns([2, 1])

# --- Prediction Logic ---
if submit_button:
    # Create a DataFrame from the user inputs
    input_data = pd.DataFrame([[
        gender, age, hypertension, heart_disease, ever_married,
        work_type, residence_type, avg_glucose_level, bmi, smoking_status
    ]], columns=['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
                 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'])

    # Encode the categorical features
    try:
        encoded_data = encode_input(input_data)
    except Exception as e:
        st.error(f"Error during data encoding. Please check your inputs. {e}")
        st.stop()
    
    # Make a prediction
    prediction = model.predict(encoded_data)[0]
    
    with col1:
        # Display the result
        st.subheader("🎯 Prediction Result")
        
        if prediction == 1:
            st.markdown("""
            <div class="prediction-box high-risk">
                <h3>⚠️ High Risk</h3>
                <p>Based on the data, the model predicts the patient is at <strong>high risk of a stroke</strong>.</p>
                <p>Please consult with a healthcare professional for proper evaluation and preventive measures.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="prediction-box low-risk">
                <h3>✅ Low Risk</h3>
                <p>Based on the data, the model predicts the patient is at <strong>low risk of a stroke</strong>.</p>
                <p>Continue maintaining healthy lifestyle habits and regular medical check-ups.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        
        # Show input data
        st.subheader("📋 Raw Input Data")
        st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
        st.dataframe(input_data, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.subheader("🔢 Processed Data for Model")
        st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
        st.dataframe(encoded_data, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.subheader("🔍 Risk Factors Analysis")
        
        # Analyze risk factors
        risk_factors = []
        if hypertension:
            risk_factors.append("• Hypertension")
        if heart_disease:
            risk_factors.append("• Heart Disease")
        if age > 65:
            risk_factors.append("• Advanced Age (>65)")
        if bmi > 30:
            risk_factors.append("• Obesity (BMI >30)")
        if avg_glucose_level > 125:
            risk_factors.append("• High Glucose (>125 mg/dL)")
        if smoking_status == "smokes":
            risk_factors.append("• Current Smoker")
        
        if risk_factors:
            st.markdown("**Identified Risk Factors:**")
            for factor in risk_factors:
                st.markdown(factor)
        else:
            st.success("✅ No major risk factors identified")
        
        # BMI Category
        st.markdown("---")
        st.subheader("📏 BMI Analysis")
        if bmi < 18.5:
            bmi_status = "Underweight"
            bmi_color = "blue"
        elif bmi < 25:
            bmi_status = "Normal"
            bmi_color = "green"
        elif bmi < 30:
            bmi_status = "Overweight"
            bmi_color = "orange"
        else:
            bmi_status = "Obese"
            bmi_color = "red"
            
        st.markdown(f"**BMI:** {bmi:.1f}")
        st.markdown(f"**Category:** <span style='color: {bmi_color}; font-weight: bold;'>{bmi_status}</span>", unsafe_allow_html=True)

else:
    with col1:
        st.info("👈 Please enter patient information in the sidebar and click 'Predict Stroke Risk' to get started.")
        
        # Educational content
        st.subheader("🧠 Understanding Stroke Risk")
        
        st.markdown("""
        **Major Risk Factors:**
        - **Age**: Risk doubles every decade after age 55
        - **High Blood Pressure**: Most important controllable risk factor
        - **Heart Disease**: Atrial fibrillation and other conditions
        - **Diabetes**: Damages blood vessels over time
        - **Smoking**: Increases risk by 2-4 times
        - **High Cholesterol**: Can block arteries to the brain
        
        **Prevention Tips:**
        - Maintain healthy blood pressure
        - Exercise regularly (30 minutes, 5 days/week)
        - Eat a balanced diet rich in fruits and vegetables
        - Don't smoke or quit if you do
        - Limit alcohol consumption
        - Manage diabetes and cholesterol
        """)
    
    with col2:
        st.subheader("🚨 Emergency Signs")
        st.markdown("""
        **F.A.S.T. Test:**
        - **F**ace: Drooping on one side
        - **A**rms: Weakness or numbness
        - **S**peech: Slurred or strange
        - **T**ime: Call 911 immediately
        
        **Other Warning Signs:**
        - Sudden confusion
        - Sudden trouble seeing
        - Sudden severe headache
        - Sudden trouble walking
        """)
        
        st.markdown("---")
        st.info("**Remember:** Time is critical in stroke treatment. Every minute counts!")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ using Streamlit")
