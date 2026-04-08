import streamlit as st
import joblib
import pandas as pd

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Kaushal AI",
    page_icon="🚀",
    layout="centered"
)

# ------------------ HEADER ------------------
st.title("🚀 Kaushal AI")
st.subheader("Career Recommendation System")

st.info("⚠️ This is a prototype demo. Predictions may not be 100% accurate.", icon="ℹ️")

# ------------------ ABOUT SECTION ------------------
with st.expander("📌 About this App"):
    st.write("""
    Kaushal AI helps you:
    - 🎯 Predict suitable career paths  
    - 📊 Analyze your skills  
    - 🧠 Get AI-based recommendations  

    Fill in your details below and discover your ideal career 🚀
    """)

# ------------------ LOAD MODELS ------------------
model = joblib.load('notebook/demo_artifacts/rf_model.pkl')
edu_interest_cert_scaler = joblib.load('notebook/demo_artifacts/edu_interest_cert_ohe.pkl')
skills_scaler = joblib.load('notebook/demo_artifacts/skills_mlb.pkl')
target_scaler = joblib.load('notebook/demo_artifacts/target_career_le.pkl')

# ------------------ FORM ------------------
with st.form("career_form"):
    st.markdown("### 🧾 Enter Your Details")

    col1, col2 = st.columns(2)

    with col1:
        education = st.selectbox(
            "🎓 Education",
            ["BCA", "BSc", "BTech", "Diploma", "MCA", "MBA"]
        )

        experience = st.slider(
            "💼 Years of Experience",
            0, 15, 1
        )

        certs = st.selectbox(
            "📜 Certification",
            ["Google Data Analytics", "AWS Certification", "Azure Certification",
             "Coursera ML", "Udemy Web Dev", "No Certification"]
        )

    with col2:
        interest = st.selectbox(
            "💡 Area of Interest",
            ["Data", "Marketing", "AI", "Web Development", "Design", "Security", "Business"]
        )

        skills = st.multiselect(
            "🛠️ Select Your Skills",
            ['AWS', 'Azure', 'Business', 'C++', 'Cloud Computing',
             'Content Writing', 'CSS', 'Cybersecurity', 'Data Analysis',
             'Deep Learning', 'Digital Marketing', 'Excel', 'Figma', 'HTML',
             'Java', 'JavaScript', 'Machine Learning', 'Networking', 'NodeJS',
             'Power BI', 'Product Management', 'Python', 'React', 'SEO', 'SQL',
             'Tableau', 'UI/UX']
        )

    submit = st.form_submit_button("🔍 Analyze Career")

# ------------------ PREDICTION ------------------
if submit:

    if len(skills) == 0:
        st.warning("⚠️ Please select at least one skill.")
    else:
        with st.spinner("Analyzing your profile... 🤖"):

            # Convert skills
            skills_list = [skill.lower() for skill in skills]
            interest_lower = interest.lower()

            input_data = {
                'education': [education],
                'experience_years': [experience],
                'interests': [interest_lower],
                'certification': [certs],
                'skills': [skills_list]
            }

            input_df = pd.DataFrame(input_data)

            # Transform
            input_skills = skills_scaler.transform(input_df["skills"])
            input_interest_cert_edu = edu_interest_cert_scaler.transform(
                input_df[['education', 'interests', 'certification']]
            )

            df_skills = pd.DataFrame(input_skills, columns=skills_scaler.classes_)
            df_interest_cert_edu = pd.DataFrame(
                input_interest_cert_edu,
                columns=edu_interest_cert_scaler.get_feature_names_out(
                    ['education', 'interests', 'certification']
                )
            )

            final_input_df = pd.concat(
                [input_df, df_skills, df_interest_cert_edu],
                axis=1
            ).drop(columns=["skills", "interests", "certification", "education"])

            # Prediction
            prediction = model.predict(final_input_df)
            predicted_career = target_scaler.inverse_transform(prediction)[0]

        # ------------------ OUTPUT ------------------
        st.success(f"🎯 Recommended Career: **{predicted_career}**")

        # Extra UX
        with st.expander("📊 View Input Summary"):
            st.write(input_df)

        st.markdown("---")
        st.markdown("💡 *Tip: Add more relevant skills to improve prediction accuracy*")