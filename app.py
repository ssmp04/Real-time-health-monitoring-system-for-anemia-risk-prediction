
import streamlit as st
import pandas as pd
import pickle

# ---------------------------
# Load Model & Encoders
# ---------------------------
model = pickle.load(open("xgboost_final_model.pkl", "rb"))
le_dict = pickle.load(open("label_encoders.pkl", "rb"))

# Default values
DEFAULT_MCH = 28.5
DEFAULT_MCHC = 33.0

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Anemia Risk Predictor", layout="centered")

st.title("Anemia Risk Prediction System")
st.write("Enter patient details to predict anemia risk level")

# ---------------------------
# Input Section
# ---------------------------
st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=0.0, step=0.1)
    mcv = st.number_input("MCV", min_value=0.0, step=0.1)

with col2:
    result = st.selectbox("Lab Result", ["Positive", "Negative"])
    diet = st.selectbox("Diet Type", ["Veg", "Non-Veg"])
    fatigue = st.selectbox("Fatigue", ["Yes", "No"])
    menstrual = st.selectbox("Menstrual Status", ["Yes", "No"])

# ---------------------------
# Optional Inputs
# ---------------------------
st.subheader("Optional Blood Parameters")

col3, col4 = st.columns(2)

with col3:
    use_mch = st.checkbox("Include MCH")
    mch = st.number_input("MCH", min_value=0.0, step=0.1) if use_mch else DEFAULT_MCH

with col4:
    use_mchc = st.checkbox("Include MCHC")
    mchc = st.number_input("MCHC", min_value=0.0, step=0.1) if use_mchc else DEFAULT_MCHC

# ---------------------------
# Preprocessing (FIXED)
# ---------------------------
def preprocess_input(data):
    df = pd.DataFrame([data])

    # Convert UI text → encoder-compatible format
    df['Gender'] = df['Gender'].replace({
        "Male": le_dict['Gender'].classes_[1],
        "Female": le_dict['Gender'].classes_[0]
    })

    df['Fatigue'] = df['Fatigue'].replace({
        "Yes": le_dict['Fatigue'].classes_[1],
        "No": le_dict['Fatigue'].classes_[0]
    })

    df['Menstrual_Status'] = df['Menstrual_Status'].replace({
        "Yes": le_dict['Menstrual_Status'].classes_[1],
        "No": le_dict['Menstrual_Status'].classes_[0]
    })

    df['Diet_Type'] = df['Diet_Type'].replace({
        "Veg": le_dict['Diet_Type'].classes_[0],
        "Non-Veg": le_dict['Diet_Type'].classes_[1]
    })

    df['Result'] = df['Result'].replace({
        "Positive": le_dict['Result'].classes_[1],
        "Negative": le_dict['Result'].classes_[0]
    })

    # Apply encoders
    categorical_cols = ['Gender', 'Diet_Type', 'Fatigue', 'Menstrual_Status', 'Result']

    for col in categorical_cols:
        df[col] = le_dict[col].transform(df[col])

    return df

# ---------------------------
# Prediction Section
# ---------------------------
st.markdown("---")

if st.button("Predict"):

    if hemoglobin == 0 or mcv == 0:
        st.error("Please enter valid medical values.")
    else:
        input_data = {
            'Gender': gender,
            'Hemoglobin': hemoglobin,
            'MCH': mch,
            'MCHC': mchc,
            'MCV': mcv,
            'Result': result,
            'Diet_Type': diet,
            'Fatigue': fatigue,
            'Menstrual_Status': menstrual
        }

        processed_data = preprocess_input(input_data)

        prediction = model.predict(processed_data)
        final_output = le_dict['Anemia_Risk_Level'].inverse_transform(prediction)

        proba = model.predict_proba(processed_data)
        confidence = max(proba[0]) * 100

        # ---------------------------
        # Output
        # ---------------------------
        st.subheader("Prediction Result")

        st.success(f"Anemia Risk Level: {final_output[0]}")
        st.info(f"Confidence Score: {confidence:.2f}%")

        if final_output[0] == "High":
            st.warning("High risk detected. Medical consultation recommended.")
        elif final_output[0] == "Medium":
            st.warning("Moderate risk. Consider monitoring and improving diet.")
        else:
            st.success("Low risk. Maintain a healthy lifestyle.")
