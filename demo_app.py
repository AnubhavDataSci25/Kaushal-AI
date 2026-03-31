import streamlit as st
import joblib
import pandas as pd

st.header("Kaushal AI - ML Model Demo", divider='rainbow')
st.subheader("Random Forest Classifier")

# Load the model and scaler
model = joblib.load('notebook/demo_artifacts/rf_model.pkl')
edu_interest_cert_scaler = joblib.load('notebook/demo_artifacts/edu_interest_cert_ohe.pkl')
skills_scaler = joblib.load('notebook/demo_artifacts/skills_mlb.pkl')
target_scaler = joblib.load('notebook/demo_artifacts/target_career_le.pkl')

# Define input fields
with st.container():
    st.subheader("Enter Your Details")
    
    education = st.selectbox("Degree*", options=["BCA", "BSc", "BTech", "Diploma", "MCA", "MBA"])
    experience = st.number_input("Years of Experience*", min_value=0, max_value=15, value=1)
    interest = st.selectbox("Area of Interest*", options=["Data", "Marketing", "AI", "Web Development", "Design", "Security", "Business"])
    skills = st.multiselect("Skills*", ['AWS', 'Azure', 'Business', 'C++', 'Cloud Computing',
            'Content Writing', 'CSS', 'Cybersecurity', 'Data Analysis',
            'Deep Learning', 'Digital Marketing', 'Excel', 'Figma', 'HTML',
            'Java', 'JavaScript', 'Machine Learning', 'Networking', 'NodeJS',
            'Power BI', 'Product Management', 'Python', 'React', 'SEO', 'SQL',
            'Tableau', 'UI/UX'])
    certs = st.selectbox("Certifications*", options=["Google Data Analytics", "AWS Certification", "Azure Certification", "Coursera ML", "Udemy Web Dev", "No Certification"])
    submit = st.button("Search")

    if submit:
        if education is None or interest is None or certs is None:
            st.error("Please fill in all required fields.")
        else:
            # Make skills into a list
            skills_list = skills if isinstance(skills, list) else [skills]

            # Make skills and interest into lower case
            skills_list = [skill.lower() for skill in skills_list]
            interest = interest.lower()

            # Create a DataFrame for the input
            input_data = {
                'education': [education],
                'experience_years': [experience],
                'interests': [interest],
                'certification': [certs],
                'skills': [skills_list]
            }

            # Convert to DataFrame
            input_df = pd.DataFrame(input_data)

            # Preprocess the input data
            input_skills = skills_scaler.transform(input_df["skills"])
            input_interest_cert_edu = edu_interest_cert_scaler.transform(input_df[['education', 'interests', 'certification']])

            # All transformed features df
            df_skills = pd.DataFrame(input_skills, columns=skills_scaler.classes_)
            df_interest_cert_edu = pd.DataFrame(input_interest_cert_edu, columns=edu_interest_cert_scaler.get_feature_names_out(['education', 'interests', 'certification']))
            final_input_df = pd.concat([input_df, df_skills, df_interest_cert_edu], axis=1).drop(columns=["skills", "interests", "certification", "education"], axis=1)

            # Show the final input DataFrame
            st.subheader("Processed Input Data")
            st.dataframe(final_input_df)

