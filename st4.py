import streamlit as st
import numpy as np
import joblib

# Load the pre-trained model
model = joblib.load('rf_model.joblib')

# Title of the app
st.title('Student Mental Health Prediction')

# Define mappings for categorical variables
ethnicity_map = {
    'Caucasian': 1, 'Asian': 2, 'Indigenous': 3, 'Hispanic or Latino': 4,
    'Black': 5, 'Other': 6, 'Metis': 7
}

sex_map = {
    'Male': 1, 'Female': 2, 'Intersex': 3
}

gender_map = {
    'Male': 1, 'Female': 2, 'Non-binary/Genderqueer': 3, 'Other': 4
}

program_map = {
    'Undergraduate': 1, 'Graduate': 2, 'Postgraduate': 3, 'PhD': 4
}

part_time_map = {
    'Yes': 1, 'No': 2
}

degree_map = {
    'None': 0, 'Associate': 1, 'Bachelor': 2, 'Master': 3, 'Doctorate': 4
}

disability_map = {
    'Yes': 1, 'No': 2
}

living_map = {
    'On-campus': 1, 'Off-campus With Family': 2, 'Off-campus with friends/roommates': 3, 'Off-campus alone': 4
}

province_map = {
    'Ontario': 1, 'Quebec': 2, 'British Columbia': 3, 'Alberta': 4,
    'Manitoba': 5, 'Other': 6
}

international_map = {
    'Yes': 1, 'No': 2
}

employment_map = {
    "Yes, full-time": 1,
    "Yes, part-time": 2,
    "No": 3
}

volunteering_map = {
    "Yes, between 1-5 hours per week": 1,
    "Yes, between 6-10 hours per week": 2,
    "Yes, between 11-15 hours per week": 3,
    "Yes, more than 15 hours per week": 4,
    "No": 5
}

plans_map = {
    "Pursue graduate studies, such as a Masters/PhD program": 1,
    "Pursue employment/Begin career": 2,
    "Pursue a professional program (e.g., law, medical, nursing, veterinary school, etc.)": 3,
    "Enter a trade (e.g., carpentry, plumbing apprenticeship, Gold Seal training, culinary arts, etc.)": 4,
    "Neither employment nor further studies (e.g., planning to be a homemaker; not planning to enter the labor market, etc.)": 5,
    "Undecided": 6
}

catch_question_map = {
    'Yes': 1, 'No': 2
}

# Define mappings for the Hobbies Importance questions
hobbies_importance_map = {
    "Hobbies_Imp_1": "How important is participating in athletics, such as varsity sports or intramurals, to you?",
    "Hobbies_Imp_2": "How important is partying or going out to bars and clubs to you?",
    "Hobbies_Imp_3": "How important is playing games, such as video games or board games, with friends to you?",
    "Hobbies_Imp_4": "How important is watching online recreational content such as on Netflix or YouTube to you?",
    "Hobbies_Imp_5": "How important is participating in academic organizations and research to you?",
    "Hobbies_Imp_6": "How important is studying to you?",
    "Hobbies_Imp_7": "How important is attending office hours to you?",
    "Hobbies_Imp_8": "How important is utilizing educational resources, such as the library and online tools, to you?"
}

# Define mappings for the Likert scale used in Hobbies Importance
likert_scale_map = {
    "Not at all important": 1,
    "Slightly important": 2,
    "Moderately important": 3,
    "Very important": 4,
    "Extremely important": 5
}

# Define mappings for the Hobbies Time questions
hobbies_time_map = {
    "Hobbies_time_1": "How many hours per week do you spend participating in athletics, such as varsity sports or intramurals?",
    "Hobbies_time_2": "How many hours per week do you spend partying or going out to bars and clubs?",
    "Hobbies_time_3": "How many hours per week do you spend playing games, such as video games or board games, with friends?",
    "Hobbies_time_4": "How many hours per week do you spend watching online recreational content such as on Netflix or YouTube?",
    "Hobbies_time_5": "How many hours per week do you spend participating in academic organizations and research?",
    "Hobbies_time_6": "How many hours per week do you spend studying?",
    "Hobbies_time_7": "How many hours per week do you spend attending office hours?",
    "Hobbies_time_8": "How many hours per week do you spend utilizing educational resources, such as the library and online tools?"
}

# Define mappings for the Likert scale used in Hobbies Time
hobbies_time_likert_scale_map = {
    "Less than 1 hour": 1,
    "1-2 hours": 2,
    "3-5 hours": 3,
    "6-10 hours": 4,
    "11-15 hours": 5,
    "16-20 hours": 6,
    "More than 20 hours": 7
}

strenuous_exercise_map = {
    0: 20, 1: 21, 2: 22, 3: 23, 4: 24, 5: 25, 6: 26, 7: 36, 8: 37, 9: 38,
    10: 39, 11: 40, 12: 41, 13: 42, 14: 43, 15: 44, 16: 45, 17: 46, 18: 47,
    19: 48, 20: 49, 21: 50, 22: 51, 23: 52, 24: 53, 25: 54, 26: 55, 27: 56,
    28: 57
}

moderate_exercise_map = {
    0: 4, 1: 5, 2: 6, 3: 7, 4: 8, 5: 9, 6: 10, 7: 11, 8: 12, 9: 13, 10: 14,
    11: 15, 12: 16, 13: 17, 14: 18, 15: 19, 16: 20, 17: 21, 18: 22, 19: 23,
    20: 24, 21: 25, 22: 26, 23: 27, 24: 28, 25: 29, 26: 30, 27: 31, 28: 32
}

mild_exercise_map = {
    0: 4, 1: 5, 2: 6, 3: 7, 4: 8, 5: 9, 6: 10, 7: 11, 8: 12, 9: 13, 10: 14,
    11: 15, 12: 16, 13: 17, 14: 18, 15: 19, 16: 20, 17: 21, 18: 22, 19: 23,
    20: 24, 21: 25, 22: 26, 23: 27, 24: 28, 25: 29, 26: 30, 27: 31, 28: 32
}

anaerobic_aerobic_map = {
    'Aerobic': 1, 'Anaerobic': 2, 'Both equally': 3
}

hours_sleep_map = {
    '4 hours or less': 1.0, '5 hours': 2.0, '6 hours': 3.0, '7 hours': 4.0,
    '8 hours': 5.0, '9 hours': 6.0, '10 hours': 7.0, '11 hours': 8.0, '12 hours or more': 9.0
}

rested_map = {
    'Yes': 1, 'Somewhat': 2, 'No': 3
}

more_sleep_map = {
    'Yes': 1, 'No, happy with their current amount of sleep': 2
}

mindfulness_freq_map = {
    'Never': 1, 'Rarely': 2, 'Sometimes': 3, 'Often': 4, 'Very often': 5
}








# Input fields
age = st.number_input('Age', min_value=0, max_value=100, value=20)
ethnicity = st.selectbox('Ethnicity', list(ethnicity_map.keys()))
sex = st.selectbox('Sex', list(sex_map.keys()))
gender = st.selectbox('Gender', list(gender_map.keys()))
year_credits = st.number_input('Academic year [1-5]', min_value=0, max_value=10, value=1)
program = st.selectbox('Program', list(program_map.keys()))
part_time = st.selectbox('Part-time (If the student is a part-time or full-time study)', list(part_time_map.keys()))

degree = st.selectbox('Current Degree', list(degree_map.keys()))
disability = st.selectbox('Disability', list(disability_map.keys()))
living = st.selectbox('Current living situation', list(living_map.keys()))
international = st.selectbox('International Student', list(international_map.keys()))
employment = st.selectbox('Employment Status', list(employment_map.keys()))
volunteering = st.selectbox('Volunteering in communities', list(volunteering_map.keys()))
plans = st.selectbox('Future Plans', list(plans_map.keys()))


# Input fields for Hobbies Importance using Likert scale
st.write(f"Hobbies Questions=\n")
st.write(f"""( 1 = Not at all important, 2 = Slightly important, 3 =
moderately important, 4 = very important, 5 = extremely important )""")

hobbies_imp = [st.slider(hobbies_importance_map[f'Hobbies_Imp_{i+1}'], min_value=1, max_value=5, value=3) for i in range(8)]


# Input fields for Hobbies Time using Likert scal
st.write(f"Hobbies Time Questions=\n")
st.write(f"""How much time the participant spent per week on
certain hobbies -\n ( 1 = Less than 1 hour, 2 = 1-2 hours, 3 = 3-5
hours, 4 = 6-10 hours, 5 = 11-15 hours, 6 = 16-20 hours, 7 = More than 20
hours ) """)

hobbies_time = [st.slider(hobbies_time_map[f'Hobbies_time_{i+1}'], min_value=1, max_value=7, value=1) for i in range(8)]



st.write('\n\n')



strenuous_exercise = st.selectbox(
    'Strenuous Exercise (times per week)',
    options=list(strenuous_exercise_map.keys()),
    format_func=lambda x: f"{x} times"
)

moderate_exercise = st.selectbox(
    'Moderate Exercise (times per week)',
    options=list(moderate_exercise_map.keys()),
    format_func=lambda x: f"{x} times"
)

mild_exercise = st.selectbox(
    'Mild Exercise (times per week)',
    options=list(mild_exercise_map.keys()),
    format_func=lambda x: f"{x} times"
)

anaerobic_aerobic = st.selectbox(
    'Anaerobic or Aerobic Exercise',
    options=list(anaerobic_aerobic_map.keys())
)




hours_sleep = st.selectbox(
    'Number of Hours Slept on an Average Night',
    options=list(hours_sleep_map.keys()),
    format_func=lambda x: x
)

rested = st.selectbox(
    'Do you feel well-rested when you wake up?',
    options=list(rested_map.keys())
)

more_sleep = st.selectbox(
    'Do you wish you were able to sleep more?',
    options=list(more_sleep_map.keys())
)

mindfulness_freq = st.selectbox(
    'How often do you engage in mindfulness or meditation?',
    options=list(mindfulness_freq_map.keys())
)





# Make a prediction
if st.button('Predict'):
    # Convert categorical inputs to numerical values
    input_data = [
        age,
        ethnicity_map[ethnicity],
        sex_map[sex],
        gender_map[gender],
        year_credits,
        year_credits,
        program_map[program],
        part_time_map[part_time],
        degree_map[degree],
        disability_map[disability],
        living_map[living],
        # Additional input fields continued
        5,  # Province is set to 5 (Ontario)
        international_map[international],
        employment_map[employment],
        volunteering_map[volunteering],
        plans_map[plans],
        *hobbies_imp,    # Exactly 8 values for hobbies importance
        *hobbies_time,   # Exactly 8 values for hobbies time
        strenuous_exercise_map[strenuous_exercise],
        moderate_exercise_map[moderate_exercise],
        mild_exercise_map[mild_exercise],
        anaerobic_aerobic_map[anaerobic_aerobic],
        hours_sleep_map[hours_sleep],
        rested_map[rested],
        more_sleep_map[more_sleep],
        mindfulness_freq_map[mindfulness_freq]
    ]

    # Ensure the input data is in the correct format for the model
    input_data = np.array(input_data).reshape(1, -1)

    # Print debugging information (optional)
    # st.write(f"Input data values: {input_data}")
    # st.write(f"Input data shape: {input_data.shape}")
    # st.write(f"Expected features: {model.n_features_in_}")
    # st.write(f"Input features: {input_data.shape[1]}")

    # Check for feature mismatch
    if input_data.shape[1] != model.n_features_in_:
        st.error(f"Feature mismatch: expected {model.n_features_in_} but got {input_data.shape[1]}")
    else:
        # Make prediction
        prediction = model.predict(input_data)
        st.write(f'Predicted Diagnosis: {"You have a mental disorder" if prediction[0] == 1 else "You do not have a mental disorder"}')

