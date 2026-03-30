import streamlit as st
import pandas as pd
import pickle
import serial
import time
import numpy as np
import csv
import os
from datetime import datetime

# ---------------- SERIAL ----------------
@st.cache_resource
def init_serial():
    try:
        s = serial.Serial('COM3', 115200, timeout=1)
        time.sleep(2)
        s.reset_input_buffer()
        return s
    except:
        return None

ser = init_serial()

if ser is None:
    st.error("Close Arduino Serial Monitor OR check COM port")
    st.stop()

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("xgboost_final_model.pkl", "rb"))
le_dict = pickle.load(open("label_encoders.pkl", "rb"))

# Default values for optional parameters (when user doesn't provide)
DEFAULT_MCH = 28.5
DEFAULT_MCHC = 33.0
DEFAULT_MCV = 90.0
DEFAULT_RESULT = 0  # Negative

# ---------------- CSV STORAGE SETUP ----------------
def init_csv_storage():
    """Create CSV file with headers if it doesn't exist"""
    filename = "patient_records.csv"
    if not os.path.exists(filename):
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Timestamp', 'Patient Name', 'Age', 'Gender', 
                'Hemoglobin (g/dL)', 'MCH (pg)', 'MCHC (g/dL)', 'MCV (fL)',
                'Diet Type', 'Fatigue', 'Menstrual Status', 'Previous Lab Result',
                'Risk Result', 'Confidence Score (%)', 'IR Value', 'Pressure Value'
            ])

def save_to_csv(patient_data):
    """Save patient data to CSV file"""
    try:
        filename = "patient_records.csv"
        with open(filename, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                patient_data['timestamp'],
                patient_data['patient_name'],
                patient_data['age'],
                patient_data['gender'],
                patient_data['hemoglobin'],
                patient_data['mch'],
                patient_data['mchc'],
                patient_data['mcv'],
                patient_data['diet'],
                patient_data['fatigue'],
                patient_data['menstrual'],
                patient_data['previous_result'],
                patient_data['risk_result'],
                patient_data['confidence_score'],
                patient_data['ir_value'],
                patient_data['pressure_value']
            ])
        return True
    except Exception as e:
        print(f"Error saving to CSV: {e}")
        return False

def get_last_patient():
    """Get the most recent patient record from CSV"""
    try:
        if os.path.exists('patient_records.csv'):
            df = pd.read_csv('patient_records.csv')
            if len(df) > 0:
                return df.iloc[-1].to_dict()
    except:
        pass
    return None

def get_all_patients():
    """Get all patient records for quick load"""
    try:
        if os.path.exists('patient_records.csv'):
            df = pd.read_csv('patient_records.csv')
            return df
    except:
        pass
    return None

# Initialize CSV storage
init_csv_storage()

# ---------------- MANUAL ENCODING MAPS ----------------
GENDER_MAP = {"Male": 1, "Female": 0}
DIET_MAP = {"Veg": 1, "Non-Veg": 0}
FATIGUE_MAP = {"Yes": 1, "No": 0}
MENSTRUAL_MAP = {"Yes": 1, "No": 0}
RESULT_MAP = {"Positive": 1, "Negative": 0}

# ---------------- GUIDANCE MESSAGES ----------------
GUIDANCE_MESSAGES = {
    "high": {
        "title": "High Risk Detected - Immediate Action Required",
        "message": """
        What this means:
        Your test results indicate a HIGH risk of anemia. This requires immediate medical attention.
        
        Recommended Actions:
        1. Visit a doctor immediately - Schedule an appointment within 24-48 hours
        2. Do not self-medicate - Iron supplements should only be taken under medical supervision
        3. Start iron-rich diet while waiting for consultation:
           - Leafy green vegetables (spinach, kale)
           - Lean red meat, poultry, fish
           - Beans, lentils, tofu
           - Fortified cereals
        4. Vitamin C rich foods to enhance iron absorption:
           - Citrus fruits, bell peppers, tomatoes
        5. Avoid tea/coffee with meals as they reduce iron absorption
        
        When to seek emergency care:
        - Severe fatigue or weakness
        - Shortness of breath
        - Chest pain or rapid heartbeat
        - Pale or yellowing skin
        """,
        "color": "error"
    },
    "moderate": {
        "title": "Moderate Risk - Medical Consultation Recommended",
        "message": """
        What this means:
        Your test results show MODERATE risk indicators for anemia. Medical consultation is recommended.
        
        Recommended Actions:
        1. Consult a doctor within 1-2 weeks for proper diagnosis
        2. Focus on iron-rich diet:
           - Include iron-rich foods in daily meals
           - Combine with Vitamin C for better absorption
           - Consider iron-fortified foods
        3. Monitor symptoms:
           - Fatigue levels
           - Skin pallor
           - Shortness of breath during exercise
        4. Lifestyle adjustments:
           - Get adequate rest
           - Stay hydrated
           - Light to moderate exercise
        
        What to discuss with your doctor:
        - Family history of anemia
        - Menstrual patterns (for women)
        - Dietary habits
        - Any ongoing medications
        """,
        "color": "warning"
    },
    "low": {
        "title": "Low Risk - Maintain Healthy Lifestyle",
        "message": """
        What this means:
        Your test results indicate LOW risk of anemia. Continue maintaining healthy habits.
        
        Recommended Actions:
        1. Maintain balanced diet:
           - Regular intake of iron-rich foods
           - Include variety of fruits and vegetables
           - Stay hydrated
        2. Healthy lifestyle:
           - Regular exercise (30 mins/day, 5 days/week)
           - Adequate sleep (7-8 hours)
           - Stress management
        3. Regular check-ups:
           - Annual health check-ups
           - Monitor any unusual symptoms
        4. Stay informed:
           - Learn about anemia risk factors
           - Share information with family/friends
        
        Preventive Tips:
        - If vegetarian, pay extra attention to iron sources
        - Women of childbearing age should monitor iron levels
        - Consider periodic hemoglobin screening
        """,
        "color": "success"
    }
}

# ---------------- CONFIDENCE SCORE CALCULATION ----------------
def calculate_confidence_score(model, df, prediction):
    """Calculate confidence score based on prediction probabilities"""
    try:
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(df)[0]
            confidence = np.max(proba) * 100
            return round(confidence, 2)
        else:
            return 85.0
    except:
        return 75.0

def get_risk_level_description(risk_level):
    """Get detailed description based on risk level"""
    descriptions = {
        "High": "Critical risk detected. Immediate medical attention required.",
        "Moderate": "Elevated risk indicators. Medical consultation recommended.",
        "Low": "Normal range. Maintain healthy lifestyle.",
        "Normal": "Normal range. Maintain healthy lifestyle."
    }
    
    for key, desc in descriptions.items():
        if key.lower() in risk_level.lower():
            return desc
    return "Risk level detected. Please consult healthcare provider."

def get_dietary_recommendations(risk_level, diet_type):
    """Get specific dietary recommendations based on risk level and diet"""
    recommendations = {
        "High": {
            "Veg": """
            Vegetarian Iron Sources (High Priority):
            - Legumes: Lentils, chickpeas, beans, soybeans
            - Dark Leafy Greens: Spinach, kale, swiss chard
            - Nuts & Seeds: Pumpkin seeds, cashews, almonds
            - Whole Grains: Quinoa, fortified cereals, oats
            - Iron-fortified foods: Breads, pastas, cereals
            
            Enhance Absorption:
            - Add Vitamin C sources (citrus, bell peppers)
            - Cook in cast-iron cookware
            - Avoid tea/coffee with meals
            """,
            "Non-Veg": """
            Animal-Based Iron Sources (High Priority):
            - Red Meat: Lean beef, lamb, organ meats (liver)
            - Poultry: Chicken, turkey (dark meat preferred)
            - Seafood: Oysters, clams, sardines, tuna
            
            Plant-Based Iron Sources:
            - Spinach, legumes, fortified cereals
            
            Enhance Absorption:
            - Combine with Vitamin C rich foods
            - Limit calcium-rich foods during iron-rich meals
            """
        },
        "Moderate": {
            "Veg": """
            Recommended Iron Sources:
            - Daily intake of legumes (1 cup)
            - Dark leafy greens in meals
            - Iron-fortified breakfast cereals
            - Pumpkin seeds as snacks
            
            Lifestyle Tips:
            - Soak legumes before cooking to enhance absorption
            - Include Vitamin C in every iron-rich meal
            """,
            "Non-Veg": """
            Recommended Iron Sources:
            - Lean red meat 2-3 times per week
            - Poultry and fish regularly
            - Include plant-based iron sources
            
            Lifestyle Tips:
            - Balance animal and plant iron sources
            - Avoid overcooking meat to preserve nutrients
            """
        },
        "Low": {
            "Veg": """
            Maintenance Diet:
            - Variety of legumes weekly
            - Regular consumption of leafy greens
            - Iron-fortified foods as needed
            - Balanced nutrition overall
            """,
            "Non-Veg": """
            Maintenance Diet:
            - Moderate red meat consumption
            - Regular fish and poultry
            - Balanced diet with variety
            """
        }
    }
    
    for level in ["High", "Moderate", "Low"]:
        if level.lower() in risk_level.lower():
            return recommendations[level].get(diet_type, recommendations[level]["Veg"])
    
    return "Consult a nutritionist for personalized dietary advice."

# ---------------- SESSION STATE INIT ----------------
for key, val in {
    "step": 0,
    "ir": 0,
    "pressure": 0,
    "hemo": 0.0,
    "detection_done": False,
    "pressure_done": False,
    "risk_result": None,
    "confidence_score": None,
    "prediction_details": None,
    "show_records": False,
    "last_patient": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Load last patient on startup
if st.session_state.last_patient is None:
    last_patient = get_last_patient()
    if last_patient:
        st.session_state.last_patient = last_patient

# ---------------- BUZZER ----------------
def trigger_buzzer():
    try:
        if ser and ser.is_open:
            ser.write(b'BUZZ\n')
            print("Buzzer triggered!")
    except Exception as e:
        print("Buzzer error:", e)

# ---------------- READ SENSOR ----------------
def read_sensor():
    if ser and ser.is_open:
        try:
            ser.reset_input_buffer()
            time.sleep(0.1)
            line = ser.readline().decode(errors='ignore').strip()
            if line:
                print("RAW:", line)
                values = line.split(",")
                if len(values) >= 2:
                    ir = float(values[0])
                    pressure = float(values[1])
                    return ir, pressure
        except Exception as e:
            print("Serial error:", e)
    return st.session_state.ir, st.session_state.pressure

# ---------------- CALIBRATION ----------------
def ir_to_hemoglobin(ir_value):
    MIN_IR = 20000
    MAX_IR = 60000
    MIN_HEMO = 8.0
    MAX_HEMO = 18.0
    ir_value = max(MIN_IR, min(ir_value, MAX_IR))
    hemo = MIN_HEMO + (ir_value - MIN_IR) / (MAX_IR - MIN_IR) * (MAX_HEMO - MIN_HEMO)
    return round(hemo, 2)

# ---------------- UI ----------------
st.title("Anemia Risk Prediction System")

# Add sidebar with instructions
with st.sidebar:
    st.header("How to Use")
    st.markdown("""
    Step-by-Step Guide:
    
    1. Prepare Device
       - Connect Arduino to USB
       - Ensure proper sensor placement
    
    2. Blood Detection
       - Place finger on IR sensor
       - Keep steady for 2 seconds
       - Wait for detection confirmation
    
    3. Pressure Test
       - Apply gentle pressure
       - Hold until measurement complete
    
    4. Patient Details
       - Fill required information
       - Optional: Add advanced lab values if available
       - Click predict for results
    
    5. Follow Guidance
       - Review risk level
       - Follow recommendations
       - Consult doctor if needed
    """)
    
    st.markdown("---")
    st.header("Normal Ranges")
    st.markdown("""
    Required:
    - Hemoglobin: 12-16 g/dL (women), 13.5-18 g/dL (men)
    
    Optional Advanced Markers:
    - MCH: 27-31 pg
    - MCHC: 32-36 g/dL
    - MCV: 80-100 fL
    """)
    
    st.markdown("---")
    st.header("View Records")
    
    if st.button("View All Patient Records"):
        if os.path.exists('patient_records.csv'):
            df = pd.read_csv('patient_records.csv')
            st.session_state.show_records = True
        else:
            st.info("No records found yet. Start making predictions!")
    
    if st.session_state.get('show_records', False):
        try:
            df = pd.read_csv('patient_records.csv')
            st.dataframe(df, use_container_width=True)
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"anemia_records_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            
            if st.button("Close Records"):
                st.session_state.show_records = False
                st.rerun()
        except:
            st.info("No records available")
    
    st.markdown("---")
    st.header("Quick Load Patient")
    
    # Quick load previous patients
    all_patients = get_all_patients()
    if all_patients is not None and len(all_patients) > 0:
        patients_list = all_patients.apply(
            lambda x: f"{x['Patient Name']} ({x['Timestamp'][:10]}) - {x['Risk Result']}", 
            axis=1
        ).tolist()
        
        selected_idx = st.selectbox(
            "Select previous patient to load:",
            range(len(patients_list)),
            format_func=lambda x: patients_list[x]
        )
        
        if st.button("Load Selected Patient"):
            st.session_state.last_patient = all_patients.iloc[selected_idx].to_dict()
            st.success(f"Loaded: {all_patients.iloc[selected_idx]['Patient Name']}")
            st.rerun()
    else:
        st.info("No previous patients found")

# Live sensor display (only when not fully done)
if st.session_state.step < 3:
    ir, pressure = read_sensor()
    st.session_state.ir = ir
    st.session_state.pressure = pressure

    st.subheader("Live Sensor Data")
    c1, c2 = st.columns(2)
    c1.metric("Blood IR", int(st.session_state.ir))
    c2.metric("Pressure", int(st.session_state.pressure))
    st.markdown("---")

# ---------------- STEP 1: START BUTTON ----------------
if st.session_state.step == 0:
    st.subheader("Step 1: Blood Detection")
    st.info("Place your finger on the IR sensor and click 'Start Detection'")
    if st.button("Start Detection"):
        st.session_state.detection_done = False
        st.session_state.pressure_done = False
        st.session_state.step = 1
        st.rerun()

# ---------------- STEP 1: BLOOD DETECTION ----------------
if st.session_state.step == 1 and not st.session_state.detection_done:
    st.subheader("Step 1: Blood Detection")
    
    with st.status("Detecting finger...", expanded=True) as status:
        st.write("Keep finger steady on the sensor")
        st.write("Reading sensor data...")
        
        time.sleep(2)
        ir, _ = read_sensor()
        st.session_state.ir = ir
        
        IR_THRESHOLD = 20000
        
        if st.session_state.ir < IR_THRESHOLD:
            status.update(label="Detection failed!", state="error")
            st.error(f"Finger NOT detected (IR={int(st.session_state.ir)}, need >{IR_THRESHOLD})")
            st.info("Tips:")
            st.info("- Place finger firmly on sensor\n- Ensure sensor is clean\n- Try repositioning finger")
            if st.button("Try Again"):
                st.session_state.step = 0
                st.rerun()
        else:
            status.update(label="Finger detected!", state="complete")
            st.session_state.hemo = ir_to_hemoglobin(st.session_state.ir)
            st.success(f"Blood detected! Hemoglobin: {st.session_state.hemo} g/dL")
            st.session_state.detection_done = True
            st.session_state.step = 2
            time.sleep(1)
            st.rerun()

# ---------------- STEP 2: PRESSURE DETECTION ----------------
if st.session_state.step == 2 and not st.session_state.pressure_done:
    st.subheader("Step 2: Pressure Detection")
    
    with st.status("Measuring pressure...", expanded=True) as status:
        st.write("Apply gentle pressure to the sensor")
        st.write("Reading pressure data...")
        
        time.sleep(2)
        _, pressure = read_sensor()
        st.session_state.pressure = pressure
        
        PRESSURE_THRESHOLD = 10
        
        if st.session_state.pressure < PRESSURE_THRESHOLD:
            status.update(label="Pressure detection failed!", state="error")
            st.error(f"Pressure NOT detected (Pressure={int(st.session_state.pressure)}, need >{PRESSURE_THRESHOLD})")
            st.info("Tips:")
            st.info("- Apply more pressure\n- Hold steady for 2 seconds\n- Try again")
            if st.button("Retry Pressure Test"):
                st.session_state.step = 1
                st.session_state.detection_done = False
                st.rerun()
        else:
            status.update(label="Pressure measured!", state="complete")
            st.success(f"Pressure detected! Reading: {int(st.session_state.pressure)}")
            st.session_state.pressure_done = True
            st.session_state.step = 3
            time.sleep(1)
            st.rerun()

# ---------------- STEP 3: PATIENT DETAILS + PREDICTION ----------------
if st.session_state.step >= 3:

    st.success("Sensors captured successfully!")
    
    # Display captured metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Hemoglobin (g/dL)", st.session_state.hemo)
        if st.session_state.hemo < 12:
            st.caption("Below normal range")
        elif st.session_state.hemo > 16:
            st.caption("Above normal range")
        else:
            st.caption("Within normal range")
    with col2:
        st.metric("Pressure Reading", int(st.session_state.pressure))
        st.caption("Pressure measurement recorded")

    st.markdown("---")
    
    # Auto-load reminder
    if st.session_state.get('last_patient'):
        st.info("Tip: Last patient data loaded automatically. You can modify or use as is.")
        
        # Button to clear loaded data
        if st.button("Clear Loaded Data"):
            st.session_state.last_patient = None
            st.rerun()
    
    # Patient Information Section with Auto-fill
    st.subheader("Patient Information")
    
    # Get last patient data for defaults
    last = st.session_state.get('last_patient', {})
    
    col1, col2, col3 = st.columns(3)
    with col1:
        patient_name = st.text_input(
            "Patient Name", 
            value=last.get('Patient Name', '') if last else '',
            placeholder="Enter name (optional)", 
            key="patient_name"
        )
    with col2:
        age = st.number_input(
            "Age", 
            min_value=0, 
            max_value=120, 
            value=int(last.get('Age', 30)) if last else 30, 
            step=1, 
            key="age"
        )
    with col3:
        contact = st.text_input(
            "Contact Number", 
            value=last.get('Contact', '') if last else '',
            placeholder="Optional", 
            key="contact"
        )
    
    st.markdown("---")
    st.subheader("Medical Information")
    
    # Required fields with auto-fill
    st.markdown("### Required Information")
    col1, col2 = st.columns(2)
    
    with col1:
        # Get last gender value
        last_gender = last.get('Gender', 'Male') if last else 'Male'
        gender = st.selectbox(
            "Gender", 
            ["Male", "Female"], 
            index=0 if last_gender == 'Male' else 1,
            key="gender"
        )
        
        # Get last diet value
        last_diet = last.get('Diet Type', 'Veg') if last else 'Veg'
        diet = st.selectbox(
            "Diet Type", 
            ["Veg", "Non-Veg"], 
            index=0 if last_diet == 'Veg' else 1,
            key="diet"
        )
        
        # Get last fatigue value
        last_fatigue = last.get('Fatigue', 'No') if last else 'No'
        fatigue = st.selectbox(
            "Fatigue (persistent tiredness)", 
            ["Yes", "No"], 
            index=0 if last_fatigue == 'Yes' else 1,
            key="fatigue"
        )
    
    with col2:
        # Get last menstrual value
        last_menstrual = last.get('Menstrual Status', 'No') if last else 'No'
        menstrual = st.selectbox(
            "Menstrual Status (for females)", 
            ["Yes", "No"], 
            index=0 if last_menstrual == 'Yes' else 1,
            key="menstrual"
        )
    
    # Optional advanced fields (collapsible)
    with st.expander("Advanced Lab Values (Optional)", expanded=False):
        st.info("If you have these values from recent blood tests, you can enter them for more accurate prediction.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            use_mch = st.checkbox("Include MCH value", key="use_mch", value=False)
            if use_mch:
                last_mch = float(last.get('MCH (pg)', 28.5)) if last else 28.5
                mch = st.number_input(
                    "MCH (pg)", 
                    min_value=20.0, 
                    max_value=40.0, 
                    value=last_mch, 
                    step=0.1, 
                    key="mch"
                )
                st.caption("Normal: 27-31 pg")
            else:
                mch = DEFAULT_MCH
        
        with col2:
            use_mchc = st.checkbox("Include MCHC value", key="use_mchc", value=False)
            if use_mchc:
                last_mchc = float(last.get('MCHC (g/dL)', 33.0)) if last else 33.0
                mchc = st.number_input(
                    "MCHC (g/dL)", 
                    min_value=28.0, 
                    max_value=40.0, 
                    value=last_mchc, 
                    step=0.1, 
                    key="mchc"
                )
                st.caption("Normal: 32-36 g/dL")
            else:
                mchc = DEFAULT_MCHC
        
        with col3:
            use_mcv = st.checkbox("Include MCV value", key="use_mcv", value=False)
            if use_mcv:
                last_mcv = float(last.get('MCV (fL)', 90.0)) if last else 90.0
                mcv = st.number_input(
                    "MCV (fL)", 
                    min_value=60.0, 
                    max_value=120.0, 
                    value=last_mcv, 
                    step=0.1, 
                    key="mcv"
                )
                st.caption("Normal: 80-100 fL")
            else:
                mcv = DEFAULT_MCV
        
        st.markdown("---")
        
        # Previous Lab Result - Optional
        use_previous_result = st.checkbox("Include Previous Lab Result", key="use_previous_result", value=False)
        if use_previous_result:
            last_result = last.get('Previous Lab Result', 'Negative') if last else 'Negative'
            previous_result = st.selectbox(
                "Previous Lab Result", 
                ["Positive", "Negative"], 
                index=0 if last_result == 'Positive' else 1,
                key="previous_result"
            )
            result_encoded = RESULT_MAP[previous_result]
            st.caption("Previous lab results help improve prediction accuracy")
        else:
            previous_result = "Not provided"
            result_encoded = DEFAULT_RESULT
        
        st.caption("If you don't provide these values, standard reference values will be used for prediction.")
    
    # Show summary of what will be used
    with st.expander("Prediction Data Summary", expanded=False):
        st.markdown("**Data that will be used for prediction:**")
        
        st.markdown("**Required:**")
        st.write(f"- Patient Name: {patient_name if patient_name else 'Anonymous'}")
        st.write(f"- Age: {age}")
        st.write(f"- Gender: {gender}")
        st.write(f"- Hemoglobin: {st.session_state.hemo} g/dL")
        st.write(f"- Diet Type: {diet}")
        st.write(f"- Fatigue: {fatigue}")
        st.write(f"- Menstrual Status: {menstrual}")
        
        st.markdown("**Optional (if provided):**")
        if use_mch:
            st.write(f"- MCH: {mch} pg (User provided)")
        else:
            st.write(f"- MCH: {DEFAULT_MCH} pg (Default value)")
        
        if use_mchc:
            st.write(f"- MCHC: {mchc} g/dL (User provided)")
        else:
            st.write(f"- MCHC: {DEFAULT_MCHC} g/dL (Default value)")
        
        if use_mcv:
            st.write(f"- MCV: {mcv} fL (User provided)")
        else:
            st.write(f"- MCV: {DEFAULT_MCV} fL (Default value)")
        
        if use_previous_result:
            st.write(f"- Previous Lab Result: {previous_result} (User provided)")
        else:
            st.write(f"- Previous Lab Result: Negative (Default value)")

    if st.button("Predict Anemia Risk", key="predict_btn", type="primary"):
        
        with st.spinner("Analyzing data and calculating risk..."):
            try:
                # Create dataframe with values (using provided or default)
                data = {
                    'Gender': GENDER_MAP[gender],
                    'Hemoglobin': st.session_state.hemo,
                    'MCH': mch if use_mch else DEFAULT_MCH,
                    'MCHC': mchc if use_mchc else DEFAULT_MCHC,
                    'MCV': mcv if use_mcv else DEFAULT_MCV,
                    'Result': result_encoded,
                    'Diet_Type': DIET_MAP[diet],
                    'Fatigue': FATIGUE_MAP[fatigue],
                    'Menstrual_Status': MENSTRUAL_MAP[menstrual]
                }
                
                df = pd.DataFrame([data])
                
                # Make prediction
                pred = model.predict(df)
                
                # Calculate confidence score
                confidence = calculate_confidence_score(model, df, pred)
                st.session_state.confidence_score = confidence
                
                # Decode prediction
                if 'Anemia_Risk_Level' not in le_dict:
                    st.error("Label encoder not found")
                    st.session_state.risk_result = None
                else:
                    encoder = le_dict['Anemia_Risk_Level']
                    predicted_value = pred[0]
                    
                    if isinstance(predicted_value, (np.integer, np.int64)):
                        predicted_value = int(predicted_value)
                    
                    result_text = encoder.inverse_transform([predicted_value])
                    
                    if isinstance(result_text, (list, np.ndarray)):
                        risk_string = str(result_text[0])
                    else:
                        risk_string = str(result_text)
                    
                    st.session_state.risk_result = risk_string
                    
                    # Store additional details
                    st.session_state.prediction_details = {
                        'gender': gender,
                        'diet': diet,
                        'fatigue': fatigue,
                        'menstrual': menstrual,
                        'hemoglobin': st.session_state.hemo,
                        'patient_name': patient_name if patient_name else "Anonymous",
                        'age': age,
                        'mch_provided': use_mch,
                        'mchc_provided': use_mchc,
                        'mcv_provided': use_mcv,
                        'result_provided': use_previous_result,
                        'mch': mch if use_mch else DEFAULT_MCH,
                        'mchc': mchc if use_mchc else DEFAULT_MCHC,
                        'mcv': mcv if use_mcv else DEFAULT_MCV,
                        'previous_result': previous_result if use_previous_result else "Not provided"
                    }
                    
                    # Save to CSV
                    patient_record = {
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'patient_name': patient_name if patient_name else "Anonymous",
                        'age': age,
                        'gender': gender,
                        'hemoglobin': st.session_state.hemo,
                        'mch': mch if use_mch else DEFAULT_MCH,
                        'mchc': mchc if use_mchc else DEFAULT_MCHC,
                        'mcv': mcv if use_mcv else DEFAULT_MCV,
                        'diet': diet,
                        'fatigue': fatigue,
                        'menstrual': menstrual,
                        'previous_result': previous_result if use_previous_result else "Not provided",
                        'risk_result': risk_string,
                        'confidence_score': confidence,
                        'ir_value': st.session_state.ir,
                        'pressure_value': st.session_state.pressure
                    }
                    
                    if save_to_csv(patient_record):
                        st.success("Patient record saved successfully!")
                        # Update last patient
                        st.session_state.last_patient = patient_record
                    else:
                        st.warning("Could not save patient record")
                    
            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.session_state.risk_result = None

    # Show result with confidence score and guidance
    if st.session_state.risk_result is not None:
        risk = str(st.session_state.risk_result)
        
        # Create expandable sections for detailed guidance
        st.markdown("---")
        
        # Main result with confidence score
        st.subheader("Prediction Result")
        
        # Determine risk category for guidance
        risk_category = "low"
        if "High" in risk:
            risk_category = "high"
        elif "Medium" in risk or "Moderate" in risk:
            risk_category = "moderate"
        
        # Display confidence score
        if st.session_state.confidence_score:
            col1, col2 = st.columns([2, 1])
            with col1:
                if risk_category == "high":
                    st.error(f"Anemia Risk Level: {risk}")
                elif risk_category == "moderate":
                    st.warning(f"Anemia Risk Level: {risk}")
                else:
                    st.success(f"Anemia Risk Level: {risk}")
                
                st.info(f"Confidence Score: {st.session_state.confidence_score}%")
                st.caption("Higher confidence scores indicate more reliable predictions")
            
            with col2:
                # Show risk meter
                risk_percentage = {
                    "high": 90,
                    "moderate": 60,
                    "low": 20
                }.get(risk_category, 50)
                st.progress(risk_percentage / 100, text=f"Risk Level: {risk_percentage}%")
        
        # Trigger buzzer for high risk
        if "High" in risk:
            trigger_buzzer()
            st.warning("Alert: Buzzer triggered due to high risk detection!")
        
        # Show detailed guidance based on risk level
        with st.expander("Medical Guidance & Recommendations", expanded=True):
            guidance = GUIDANCE_MESSAGES.get(risk_category, GUIDANCE_MESSAGES["low"])
            
            if risk_category == "high":
                st.error(guidance["title"])
            elif risk_category == "moderate":
                st.warning(guidance["title"])
            else:
                st.success(guidance["title"])
            
            st.markdown(guidance["message"])
        
        # Dietary recommendations
        with st.expander("Dietary Recommendations", expanded=True):
            diet_type = st.session_state.prediction_details['diet']
            dietary_advice = get_dietary_recommendations(risk, diet_type)
            st.markdown(dietary_advice)
        
        # Risk factors and monitoring
        with st.expander("Understanding Your Results"):
            st.markdown(f"""
            What the {risk} risk means:
            {get_risk_level_description(risk)}
            
            Patient Information:
            - Patient Name: {st.session_state.prediction_details['patient_name']}
            - Age: {st.session_state.prediction_details['age']}
            - Gender: {st.session_state.prediction_details['gender']}
            
            Key Factors Considered:
            - Hemoglobin: {st.session_state.prediction_details['hemoglobin']} g/dL 
              (Normal: 12-16 g/dL for women, 13.5-18 g/dL for men)
            - Diet Type: {st.session_state.prediction_details['diet']}
            - Fatigue: {st.session_state.prediction_details['fatigue']}
            - Menstrual Status: {st.session_state.prediction_details['menstrual']}
            """)
            
            # Show optional values if provided
            if st.session_state.prediction_details['mch_provided'] or \
               st.session_state.prediction_details['mchc_provided'] or \
               st.session_state.prediction_details['mcv_provided'] or \
               st.session_state.prediction_details['result_provided']:
                
                st.markdown("**Additional Data Used (if provided):**")
                
                if st.session_state.prediction_details['mch_provided']:
                    st.write(f"- MCH: {st.session_state.prediction_details['mch']} pg (Normal: 27-31 pg)")
                if st.session_state.prediction_details['mchc_provided']:
                    st.write(f"- MCHC: {st.session_state.prediction_details['mchc']} g/dL (Normal: 32-36 g/dL)")
                if st.session_state.prediction_details['mcv_provided']:
                    st.write(f"- MCV: {st.session_state.prediction_details['mcv']} fL (Normal: 80-100 fL)")
                if st.session_state.prediction_details['result_provided']:
                    st.write(f"- Previous Lab Result: {st.session_state.prediction_details['previous_result']}")
            
            st.markdown("""
            Next Steps:
            1. Review the recommendations above
            2. Schedule follow-up with healthcare provider if needed
            3. Track symptoms and diet for 2 weeks
            4. Consider retesting if symptoms persist
            """)
        
        # When to seek help
        if risk_category in ["high", "moderate"]:
            with st.expander("When to Seek Immediate Medical Help"):
                st.warning("""
                Seek immediate medical attention if you experience:
                - Severe fatigue or weakness interfering with daily activities
                - Shortness of breath at rest or with minimal activity
                - Rapid or irregular heartbeat
                - Chest pain
                - Fainting or dizziness
                - Pale or yellowing skin
                - Cold hands and feet
                """)

    st.markdown("---")
    
    # Reset button with confirmation
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Start New Test", key="reset_btn", type="secondary"):
            for key in ["step", "ir", "pressure", "hemo",
                        "detection_done", "pressure_done", "risk_result",
                        "confidence_score", "prediction_details"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
