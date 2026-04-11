from flask import Flask, render_template, request
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

app = Flask(__name__)

EDUCATION_OPTIONS = ["BCA", "BSc", "BTech", "Diploma", "MBA", "MCA"]

INTEREST_OPTIONS = [
    ("ai", "Artificial Intelligence"),
    ("business", "Business"),
    ("data", "Data Science"),
    ("design", "Design"),
    ("marketing", "Marketing"),
    ("security", "Cybersecurity"),
    ("web development", "Web Development"),
]

CERTIFICATION_OPTIONS = [
    ("NA", "No Certification"),
    ("AWS Certification", "AWS Certification"),
    ("Azure Fundamentals", "Azure Fundamentals"),
    ("Coursera ML", "Coursera ML"),
    ("Google Data Analytics", "Google Data Analytics"),
    ("Udemy Web Dev", "Udemy Web Dev"),
]

SKILLS_OPTIONS = [
    "aws", "azure", "business", "c++", "cloud computing", "content writing",
    "css", "cybersecurity", "data analysis", "deep learning", "digital marketing",
    "excel", "figma", "html", "java", "javascript", "machine learning",
    "networking", "nodejs", "power bi", "product management", "python",
    "react", "seo", "sql", "tableau", "ui/ux",
]


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    form_data = {}

    if request.method == "POST":
        try:
            education = request.form.get("education", "").strip()
            experience_years = int(request.form.get("experience_years", 0))
            interests = request.form.get("interests", "").strip()
            certification = request.form.get("certification", "NA").strip()
            skills = request.form.getlist("skills")

            form_data = {
                "education": education,
                "experience_years": experience_years,
                "interests": interests,
                "certification": certification,
                "skills": skills,
            }

            if not skills:
                error = "Please select at least one skill."
            else:
                data = CustomData(
                    education=education,
                    experience_years=experience_years,
                    skills=skills,
                    interests=interests,
                    certifications=certification,
                )
                df = data.get_data_as_df()
                pipeline = PredictPipeline()
                result = pipeline.predict(df)
                prediction = result[0]
        except Exception as e:
            error = f"Prediction failed: {str(e)}"

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        form_data=form_data,
        education_options=EDUCATION_OPTIONS,
        interest_options=INTEREST_OPTIONS,
        certification_options=CERTIFICATION_OPTIONS,
        skills_options=SKILLS_OPTIONS,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
