# -*- coding: utf-8 -*-

import streamlit as st
import PyPDF2
import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

st.set_page_config(layout="wide")
st.markdown("### Created by: Satyush Mohapatra")
st.markdown("---")

SKILL_MAP = {
    "oop": [
        "oop", "oops", "object oriented programming",
        "object-oriented programming", "encapsulation",
        "inheritance", "polymorphism", "abstraction"
    ],
    "dp": ["dynamic programming"],
    "ml": ["machine learning"],
    "dl": ["deep learning"],
    "nlp": ["natural language processing"],
    "ai": ["artificial intelligence"],
    "dbms": ["database management system"],
    "dsa": ["data structures", "algorithms"],
    "js": ["javascript"],
    "react": ["reactjs", "react js", "react.js"],
    "node": ["nodejs", "node js"],
    "sql": ["mysql", "postgresql", "sqlite"],
    "os": ["operating system", "operating systems"],
    "cn": ["computer networks", "networking"]
}

SKILLS_DB = {
    "Data Scientist": ["python", "ml", "pandas", "numpy", "statistics"],
    "Web Developer": ["html", "css", "js", "react", "node"],
    "Android Developer": ["java", "kotlin", "android", "firebase", "oop"],
    "DevOps Engineer": ["docker", "kubernetes", "aws", "ci cd"],
    "Cyber Security": ["network security", "penetration testing", "encryption"],
    "UI/UX Designer": ["figma", "wireframe", "prototype", "design"],
    "General Professional": ["communication", "teamwork", "leadership"]
}

@st.cache_resource
def load_model():
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

model = load_model()

def extract_text(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

def preprocess(text):
    text = text.lower()

    text = text.replace("-", " ")
    text = text.replace(".", " ")

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    for skill, variants in SKILL_MAP.items():
        for v in variants:
            pattern = r"\b" + re.escape(v) + r"\b"
            text = re.sub(pattern, skill, text)

    return text

def detect_skills(resume_text, job_skills):
    clean_text = preprocess(resume_text)

    found = []
    missing = []

    for skill in job_skills:
        skill = skill.lower()

        if skill in clean_text:
            found.append(skill)
        else:
            missing.append(skill)

    return found, missing, clean_text

st.title("Resume Screening System")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf"])

if uploaded_file:

    resume_text = extract_text(uploaded_file)

    if not resume_text.strip():
        st.error("Could not read resume properly")
        st.stop()

    st.write("### Resume Preview")
    st.write(resume_text[:300])

    resume_clean = preprocess(resume_text)

    if model is None:
        st.error("Model not loaded properly")
        st.stop()

    try:
        resume_embedding = model.encode(resume_clean)
    except Exception as e:
        st.error(f"Encoding failed: {e}")
        st.stop()

    scores = {}

    for role, skills in SKILLS_DB.items():

        job_text = " ".join(skills)

        try:
            job_embedding = model.encode(job_text)
        except Exception as e:
            st.error(f"Job encoding failed: {e}")
            st.stop()

        score = cosine_similarity(
            [resume_embedding],
            [job_embedding]
        )[0][0]

        scores[role] = score

    sorted_scores = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    best_role = None
    threshold = 0.2

    if sorted_scores and sorted_scores[0][1] > threshold:
        best_role = sorted_scores[0][0]

    st.header("Best Role")

    if best_role:
        st.success(f"Best match: {best_role}")
    else:
        st.warning("No strong match found")

    st.header("Top 3 Recommended Roles")

    for i, (role, score) in enumerate(sorted_scores[:3], 1):
        st.write(f"{i}. {role} ({round(score * 100, 2)}%)")

    st.header("Ranking")

    for i, (role, score) in enumerate(sorted_scores, 1):
        normalized_score = (score + 1) / 2
        progress_value = int(normalized_score * 100)

        st.progress(progress_value)
        st.write(f"{i}. {role} ({round(score * 100, 2)}%)")

    st.header("Missing Skills")

    if best_role and best_role in SKILLS_DB:

        job_skills = SKILLS_DB[best_role]

        found_skills, missing_skills, _ = detect_skills(resume_text, job_skills)

        if missing_skills:
            st.warning(", ".join(missing_skills))
        else:
            st.success("No missing skills")
