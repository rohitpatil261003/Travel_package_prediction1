# import streamlit as st
# import numpy as np
# import joblib

# model=joblib.load('model.joblib')

# # Page Title

# st.set_page_config(
#     page_title="Travel Package Prediction",
#     layout="centered"
# )

# st.title("Travel Package Prediction")

# loaded_model=joblib.load("model.joblib")
# loaded_preprocessor=joblib.load("preprocessor.joblib")

import streamlit as st
import numpy as np
import joblib

# --------------------------------------------------
# Page Setup
# --------------------------------------------------
st.set_page_config(
    page_title="Holiday Package Purchase Predictor",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Tourism Package Purchase Prediction")
st.write("Enter customer attributes below to predict whether they are likely to purchase the pitched holiday package.")

# --------------------------------------------------
# Load Pre-trained Artifacts
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load('preprocessor.joblib')
    models_dict = joblib.load('model.joblib')
    # Extracts the fitted model instance (XGBClassifier) saved inside the dictionary
    model = list(models_dict.values())[0] if isinstance(models_dict, dict) else models_dict
    return preprocessor, model

try:
    preprocessor, model = load_artifacts()
except Exception as e:
    st.error(f"Error loading artifacts: {e}. Please ensure 'preprocessor.joblib' and 'model.joblib' are in the working directory.")
    st.stop()

# --------------------------------------------------
# Streamlit Input Features Layout
# --------------------------------------------------
with st.form("prediction_form"):
    st.subheader("Customer Demographics & Work Profile")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
        gender = st.selectbox("Gender", options=["Male", "Female"])
        marital_status = st.selectbox("Marital Status", options=["Married", "Unmarried", "Divorced"])

    with col2:
        occupation = st.selectbox("Occupation", options=["Salaried", "Small Business", "Large Business", "Free Lancer"])
        designation = st.selectbox("Designation", options=["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        monthly_income = st.number_input("Monthly Income (₹)", min_value=1000.0, max_value=200000.0, value=22000.0, step=500.0)

    with col3:
        city_tier = st.selectbox("City Tier", options=[1, 2, 3], index=0)
        passport = st.selectbox("Holds Passport?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        own_car = st.selectbox("Owns a Car?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    st.subheader("Trip & Interaction Details")
    col4, col5, col6 = st.columns(3)

    with col4:
        typeof_contact = st.selectbox("Type of Contact", options=["Self Enquiry", "Company Invited"])
        product_pitched = st.selectbox("Product Pitched", options=["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
        preferred_property_star = st.slider("Preferred Property Star Rating", min_value=1.0, max_value=5.0, value=3.0, step=1.0)

    with col5:
        num_trips = st.number_input("Average Number of Annual Trips", min_value=0.0, max_value=50.0, value=3.0, step=1.0)
        num_person_visiting = st.number_input("Number of Adults Visiting", min_value=1, max_value=10, value=2, step=1)
        num_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=1, step=1)

    with col6:
        duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1.0, max_value=150.0, value=15.0, step=1.0)
        num_followups = st.slider("Number of Follow-ups", min_value=1.0, max_value=10.0, value=3.0, step=1.0)
        pitch_satisfaction_score = st.slider("Pitch Satisfaction Score (1 to 5)", min_value=1, max_value=5, value=3, step=1)

    submitted = st.form_submit_button("Predict Package Purchase")

# --------------------------------------------------
# Data Transformation & Prediction Logic
# --------------------------------------------------
if submitted:
    # 1. Feature Engineering matching the training notebook
    total_visiting = float(num_person_visiting + num_children_visiting)

    # 2. Assemble DataFrame adhering to training feature names and ordering
    raw_input_df = pd.DataFrame([{
        'Age': float(age),
        'TypeofContact': typeof_contact,
        'CityTier': int(city_tier),
        'DurationOfPitch': float(duration_of_pitch),
        'Occupation': occupation,
        'Gender': gender,
        'NumberOfFollowups': float(num_followups),
        'ProductPitched': product_pitched,
        'PreferredPropertyStar': float(preferred_property_star),
        'MaritalStatus': marital_status,
        'NumberOfTrips': float(num_trips),
        'Passport': int(passport),
        'PitchSatisfactionScore': int(pitch_satisfaction_score),
        'OwnCar': int(own_car),
        'Designation': designation,
        'MonthlyIncome': float(monthly_income),
        'Total_Visiting': total_visiting
    }])

    # 3. Transform features using saved ColumnTransformer
    transformed_features = preprocessor.transform(raw_input_df)

    # 4. Generate model prediction and probability score
    prediction = model.predict(transformed_features)[0]
    prob = model.predict_proba(transformed_features)[0][1]

    # ----------------------------------------------
    # Results Display
    # ----------------------------------------------
    st.divider()
    if prediction == 1:
        st.success(f"**Likely to Purchase!** (Confidence: {prob * 100:.2f}%)")
        st.balloons()
    else:
        st.warning(f"**Unlikely to Purchase.** (Purchase Probability: {prob * 100:.2f}%)")