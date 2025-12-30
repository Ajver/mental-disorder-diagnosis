import streamlit as st
import requests

st.set_page_config(page_title="Mental Health Survey", layout="centered")

st.title("Mental Health Survey")

scale_1_4 = {
    "Seldom": 1,
    "Sometimes": 2,
    "Usually": 3,
    "Most-Often": 4,
}

yes_no = {
    "No": 0,
    "Yes": 1,
}

sadness = st.selectbox(
    "1. How often do you feel sad or low?",
    options=list(scale_1_4.keys())
)

euphoric = st.selectbox(
    "2. How often do you feel unusually happy or euphoric?",
    options=list(scale_1_4.keys())
)

exhausted = st.selectbox(
    "3. How often do you feel physically or mentally exhausted?",
    options=list(scale_1_4.keys())
)

sleep_disorder = st.selectbox(
    "4. How often do you experience problems with sleep (difficulty falling asleep, staying asleep, or oversleeping)?",
    options=list(scale_1_4.keys())
)

mood_swing = st.radio(
    "5. Do you experience frequent changes in your mood without a clear reason?",
    options=list(yes_no.keys()),
    horizontal=True
)

suicidal = st.radio(
    "6. Have you had thoughts about harming yourself or ending your life?",
    options=list(yes_no.keys()),
    horizontal=True
)

anorexia = st.radio(
    "7. Have you experienced a significant fear of gaining weight or intentionally restricted your eating?",
    options=list(yes_no.keys()),
    horizontal=True
)

authority = st.radio(
    "8. Do you generally respect and follow instructions from authority figures?",
    options=list(yes_no.keys()),
    horizontal=True
)

aggressive = st.radio(
    "9. Do you often respond aggressively when you feel stressed or provoked?",
    options=list(yes_no.keys()),
    horizontal=True
)

nervous = st.radio(
    "10. Have you ever experienced a nervous or emotional breakdown?",
    options=list(yes_no.keys()),
    horizontal=True
)

overthinking = st.radio(
    "11. Do you tend to overthink situations or decisions?",
    options=list(yes_no.keys()),
    horizontal=True
)

try_explain = st.radio(
    "12. When conflicts arise, do you usually try to explain your point of view calmly?",
    options=list(yes_no.keys()),
    horizontal=True
)

ignore_move_on = st.radio(
    "13. Are you able to ignore negative situations and move on without dwelling on them?",
    options=list(yes_no.keys()),
    horizontal=True
)

admit_mistakes = st.radio(
    "14. Do you usually admit your mistakes when you realize you are wrong?",
    options=list(yes_no.keys()),
    horizontal=True
)

sexual_activity = st.slider(
    "15. On a scale of 1 to 10, how would you rate your level of sexual activity?",
    min_value=1,
    max_value=10,
    value=5
)

concentration = st.slider(
    "16. On a scale of 1 to 10, how would you rate your ability to concentrate?",
    min_value=1,
    max_value=10,
    value=5
)

optimism = st.slider(
    "13. On a scale of 1 to 10, how optimistic do you generally feel about your future?",
    min_value=1,
    max_value=10,
    value=5
)

st.divider()

if st.button("Diagnose me"):
    payload = {
        "data": {
            "Sadness": scale_1_4[sadness],
            "Euphoric": scale_1_4[euphoric],
            "Exhausted": scale_1_4[exhausted],
            "Sleep dissorder": scale_1_4[sleep_disorder],
            "Mood Swing": yes_no[mood_swing],
            "Suicidal thoughts": yes_no[suicidal],
            "Anorxia": yes_no[anorexia],
            "Authority Respect": yes_no[authority],
            "Try-Explanation": yes_no[try_explain],
            "Aggressive Response": yes_no[aggressive],
            "Ignore & Move-On": yes_no[ignore_move_on],
            "Nervous Break-down": yes_no[nervous],
            "Admit Mistakes": yes_no[admit_mistakes],
            "Overthinking": yes_no[overthinking],
            "Sexual Activity": sexual_activity,
            "Concentration": concentration,
            "Optimism": optimism,
        }
    }

    try:
        response = requests.post(
            "http://0.0.0.0:8000/predict",
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        result = response.json()

        if result.get("success"):
            diagnose = result["prediction"]["diagnose"]
            st.success(f"**Diagnosis:** {diagnose}")
        else:
            st.error("Prediction failed.")

    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to prediction service.\n\n{e}")
