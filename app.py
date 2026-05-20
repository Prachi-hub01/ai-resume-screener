"""
AI Resume Screener — Smart Resume-Job Matching with NLP
Built by Prachi Arvind Dutt
"""

import streamlit as st
import os
import re
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import fitz  # PyMuPDF
from collections import Counter

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');
    
    * { font-family: 'DM Sans', sans-serif; }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #1e3a5f, #2d8cf0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .score-card {
        background: linear-gradient(135deg, #1e3a5f, #2d8cf0);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(30, 58, 95, 0.3);
    }
    
    .score-number {
        font-size: 4rem;
        font-weight: 700;
        line-height: 1;
    }
    
    .score-label {
        font-size: 1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    .metric-card {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #2d8cf0;
        margin-bottom: 1rem;
    }
    
    .skill-match { color: #10b981; font-weight: 600; }
    .skill-missing { color: #ef4444; font-weight: 600; }
    .skill-extra { color: #f59e0b; font-weight: 600; }
    
    .suggestion-box {
        background: #fffbeb;
        border: 1px solid #fbbf24;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    
    .strength-box {
        background: #ecfdf5;
        border: 1px solid #10b981;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Skills Database
# ──────────────────────────────────────────────
TECH_SKILLS = {
    # Programming Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "ruby", "go", "golang",
    "rust", "swift", "kotlin", "php", "scala", "r", "matlab", "perl", "dart", "lua",
    "html", "css", "sass", "less", "sql", "nosql", "bash", "shell", "powershell",
    
    # Frameworks & Libraries
    "react", "reactjs", "react.js", "angular", "angularjs", "vue", "vuejs", "vue.js",
    "next.js", "nextjs", "nuxt.js", "svelte", "django", "flask", "fastapi",
    "spring", "spring boot", "springboot", "express", "expressjs", "node.js", "nodejs",
    "asp.net", ".net", "dotnet", "laravel", "rails", "ruby on rails",
    "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn", "pandas", "numpy",
    "matplotlib", "seaborn", "plotly", "streamlit", "gradio", "langchain",
    "hugging face", "huggingface", "transformers", "opencv", "spacy", "nltk",
    
    # Cloud & DevOps
    "aws", "amazon web services", "azure", "gcp", "google cloud", "heroku", "vercel",
    "docker", "kubernetes", "k8s", "jenkins", "ci/cd", "terraform", "ansible",
    "github actions", "gitlab ci", "circleci", "nginx", "apache",
    
    # Databases
    "mysql", "postgresql", "postgres", "mongodb", "redis", "elasticsearch",
    "dynamodb", "cassandra", "sqlite", "oracle", "firebase", "supabase",
    "neo4j", "graphql", "prisma", "sequelize", "sqlalchemy",
    
    # AI/ML Concepts
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "reinforcement learning", "generative ai", "gen ai",
    "llm", "large language models", "rag", "retrieval augmented generation",
    "fine-tuning", "transfer learning", "neural networks", "cnn", "rnn", "lstm",
    "transformer", "bert", "gpt", "diffusion models", "gan",
    
    # Data & Analytics
    "data science", "data analysis", "data engineering", "etl", "data pipeline",
    "power bi", "tableau", "looker", "apache spark", "hadoop", "airflow",
    "snowflake", "databricks", "bigquery", "redshift", "kafka",
    
    # Tools & Practices
    "git", "github", "gitlab", "bitbucket", "jira", "confluence", "figma",
    "agile", "scrum", "kanban", "tdd", "test driven development",
    "rest api", "restful", "microservices", "api design", "swagger",
    "linux", "unix", "windows server",
    
    # Security
    "cybersecurity", "penetration testing", "oauth", "jwt", "encryption",
    "ssl", "tls", "sso", "identity management",
}

SOFT_SKILLS = {
    "leadership", "communication", "teamwork", "problem solving", "problem-solving",
    "critical thinking", "time management", "project management", "collaboration",
    "mentoring", "coaching", "presentation", "public speaking", "negotiation",
    "stakeholder management", "cross-functional", "strategic thinking", "analytical",
    "creative thinking", "adaptability", "decision making", "conflict resolution",
    "customer focus", "attention to detail", "self-motivated", "proactive",
}

EDUCATION_KEYWORDS = {
    "bachelor", "master", "phd", "doctorate", "mba", "bsc", "msc", "b.tech", "m.tech",
    "b.e.", "m.e.", "diploma", "certification", "certified", "degree", "university",
    "college", "institute", "academy",
}

EXPERIENCE_PATTERNS = [
    r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of)?\s*(?:experience|exp)',
    r'(?:experience|exp)\s*(?:of)?\s*(\d+)\+?\s*(?:years?|yrs?)',
]


# ──────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the sentence transformer model."""
    return SentenceTransformer('all-MiniLM-L6-v2')


def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file."""
    try:
        pdf_bytes = pdf_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""


def clean_text(text):
    """Clean and normalize text."""
    text = text.lower()
    text = re.sub(r'[^\w\s\-\+\#\.]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_skills(text, skill_set):
    """Extract skills from text."""
    text_lower = text.lower()
    found_skills = set()
    for skill in skill_set:
        # Use word boundary matching for short skills
        if len(skill) <= 3:
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower):
                found_skills.add(skill)
        else:
            if skill in text_lower:
                found_skills.add(skill)
    return found_skills


def extract_experience_years(text):
    """Extract years of experience from text."""
    text_lower = text.lower()
    years = []
    for pattern in EXPERIENCE_PATTERNS:
        matches = re.findall(pattern, text_lower)
        years.extend([int(y) for y in matches])
    return max(years) if years else None


def extract_education(text):
    """Extract education-related information."""
    text_lower = text.lower()
    found_education = set()
    for keyword in EDUCATION_KEYWORDS:
        if keyword in text_lower:
            found_education.add(keyword)
    return found_education


def compute_semantic_similarity(model, text1, text2):
    """Compute semantic similarity between two texts using sentence transformers."""
    embeddings = model.encode([text1, text2])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return float(similarity)


def compute_tfidf_similarity(text1, text2):
    """Compute TF-IDF based similarity."""
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return float(similarity)


def extract_keywords_tfidf(text, n=20):
    """Extract top keywords using TF-IDF."""
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]
    keyword_scores = list(zip(feature_names, scores))
    keyword_scores.sort(key=lambda x: x[1], reverse=True)
    return keyword_scores[:n]


def compute_overall_score(semantic_sim, tfidf_sim, skill_match_pct, has_education_match):
    """Compute weighted overall match score."""
    education_bonus = 5 if has_education_match else 0
    score = (
        semantic_sim * 35 +
        tfidf_sim * 25 +
        skill_match_pct * 35 +
        education_bonus
    )
    return min(round(score, 1), 100)


def get_score_rating(score):
    """Get rating label and color based on score."""
    if score >= 85:
        return "Excellent Match", "🟢"
    elif score >= 70:
        return "Strong Match", "🔵"
    elif score >= 55:
        return "Good Match", "🟡"
    elif score >= 40:
        return "Fair Match", "🟠"
    else:
        return "Weak Match", "🔴"


def generate_suggestions(matched_skills, missing_skills, extra_skills, 
                          resume_exp_years, jd_exp_years, score):
    """Generate improvement suggestions."""
    suggestions = []
    strengths = []
    
    # Skill-based suggestions
    if missing_skills:
        top_missing = list(missing_skills)[:5]
        suggestions.append(
            f"**Add missing key skills:** {', '.join(top_missing)}. "
            f"These are mentioned in the job description but not in your resume."
        )
    
    if len(matched_skills) > 5:
        strengths.append(
            f"**Strong skill alignment:** You match {len(matched_skills)} skills "
            f"with the job requirements — {', '.join(list(matched_skills)[:6])}."
        )
    
    if extra_skills:
        top_extra = list(extra_skills)[:5]
        strengths.append(
            f"**Bonus skills:** You have additional skills ({', '.join(top_extra)}) "
            f"that could add value beyond the core requirements."
        )
    
    # Experience-based
    if resume_exp_years and jd_exp_years:
        if resume_exp_years >= jd_exp_years:
            strengths.append(
                f"**Experience level:** Your {resume_exp_years} years of experience "
                f"meets or exceeds the required {jd_exp_years} years."
            )
        else:
            suggestions.append(
                f"**Experience gap:** The role requires {jd_exp_years} years, "
                f"but your resume shows {resume_exp_years} years. Highlight transferable "
                f"experience, internships, or relevant projects to bridge this gap."
            )
    
    # General suggestions based on score
    if score < 55:
        suggestions.append(
            "**Tailor your resume:** Rewrite your summary/objective to mirror "
            "the key phrases and requirements in the job description."
        )
        suggestions.append(
            "**Use matching keywords:** ATS systems scan for exact keyword matches. "
            "Incorporate the job description's terminology into your bullet points."
        )
    
    if score >= 70:
        strengths.append(
            "**High overall compatibility:** Your profile strongly aligns with "
            "this role. Focus on customizing your cover letter and preparing "
            "for interviews on the matched topics."
        )
    
    return suggestions, strengths


# ──────────────────────────────────────────────
# Main App
# ──────────────────────────────────────────────
def main():
    # Header
    st.markdown('<div class="main-header">📄 AI Resume Screener</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Smart Resume-Job Matching powered by NLP & Sentence Transformers</div>',
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ About")
        st.markdown("""
        **AI Resume Screener** uses NLP to analyze how well your resume matches a job description.
        
        **How it works:**
        - 🔍 Semantic similarity (Sentence-BERT)
        - 📊 TF-IDF keyword matching
        - 🎯 Skill extraction & comparison
        - 💡 Actionable improvement tips
        """)
        
        st.divider()
        st.markdown("**Model:** all-MiniLM-L6-v2")
        st.markdown("**Methods:** SBERT + TF-IDF + Skill Matching")
        st.markdown("**Built by:** Prachi Arvind Dutt")
    
    # Input Section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📄 Upload Resume")
        resume_file = st.file_uploader(
            "Upload your resume (PDF)",
            type=["pdf"],
            key="resume"
        )
        resume_text_input = st.text_area(
            "Or paste resume text here:",
            height=200,
            placeholder="Paste your resume content here..."
        )
    
    with col2:
        st.markdown("### 💼 Job Description")
        jd_text = st.text_area(
            "Paste the job description:",
            height=280,
            placeholder="Paste the full job description here..."
        )
    
    # Process
    if st.button("🔍 Analyze Match", type="primary", use_container_width=True):
        # Get resume text
        resume_text = ""
        if resume_file:
            with st.spinner("Reading PDF..."):
                resume_text = extract_text_from_pdf(resume_file)
        elif resume_text_input:
            resume_text = resume_text_input
        
        if not resume_text:
            st.error("Please upload a resume PDF or paste resume text.")
            return
        if not jd_text:
            st.error("Please paste a job description.")
            return
        
        # Analysis
        with st.spinner("🔍 Analyzing your resume against the job description..."):
            model = load_model()
            
            # Clean texts
            resume_clean = clean_text(resume_text)
            jd_clean = clean_text(jd_text)
            
            # 1. Semantic Similarity
            semantic_sim = compute_semantic_similarity(model, resume_clean, jd_clean)
            
            # 2. TF-IDF Similarity
            tfidf_sim = compute_tfidf_similarity(resume_clean, jd_clean)
            
            # 3. Skill Extraction
            resume_tech_skills = extract_skills(resume_text, TECH_SKILLS)
            jd_tech_skills = extract_skills(jd_text, TECH_SKILLS)
            resume_soft_skills = extract_skills(resume_text, SOFT_SKILLS)
            jd_soft_skills = extract_skills(jd_text, SOFT_SKILLS)
            
            all_resume_skills = resume_tech_skills | resume_soft_skills
            all_jd_skills = jd_tech_skills | jd_soft_skills
            
            matched_skills = all_resume_skills & all_jd_skills
            missing_skills = all_jd_skills - all_resume_skills
            extra_skills = all_resume_skills - all_jd_skills
            
            skill_match_pct = (len(matched_skills) / len(all_jd_skills) * 100) if all_jd_skills else 0
            
            # 4. Experience
            resume_exp = extract_experience_years(resume_text)
            jd_exp = extract_experience_years(jd_text)
            
            # 5. Education
            resume_education = extract_education(resume_text)
            jd_education = extract_education(jd_text)
            education_match = bool(resume_education & jd_education)
            
            # 6. Overall Score
            overall_score = compute_overall_score(
                semantic_sim, tfidf_sim, skill_match_pct / 100, education_match
            )
            
            rating, emoji = get_score_rating(overall_score)
            
            # 7. Suggestions
            suggestions, strengths = generate_suggestions(
                matched_skills, missing_skills, extra_skills,
                resume_exp, jd_exp, overall_score
            )
            
            # 8. Keywords
            jd_keywords = extract_keywords_tfidf(jd_text, n=15)
        
        # ──────────────────────────────────────
        # Results Display
        # ──────────────────────────────────────
        st.divider()
        st.markdown("## 📊 Analysis Results")
        
        # Score Card
        r1, r2, r3 = st.columns([1, 2, 1])
        with r2:
            st.markdown(f"""
            <div class="score-card">
                <div class="score-number">{overall_score}%</div>
                <div class="score-label">{emoji} {rating}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("")
        
        # Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Semantic Match", f"{semantic_sim*100:.1f}%")
        with m2:
            st.metric("Keyword Match", f"{tfidf_sim*100:.1f}%")
        with m3:
            st.metric("Skills Match", f"{skill_match_pct:.0f}%")
        with m4:
            st.metric("Skills Found", f"{len(matched_skills)}/{len(all_jd_skills)}")
        
        st.divider()
        
        # Skills Breakdown
        st.markdown("### 🎯 Skills Analysis")
        
        s1, s2, s3 = st.columns(3)
        
        with s1:
            st.markdown("**✅ Matched Skills**")
            if matched_skills:
                for skill in sorted(matched_skills):
                    st.markdown(f'<span class="skill-match">✓ {skill}</span>', unsafe_allow_html=True)
            else:
                st.markdown("*No matching skills found*")
        
        with s2:
            st.markdown("**❌ Missing Skills**")
            if missing_skills:
                for skill in sorted(missing_skills):
                    st.markdown(f'<span class="skill-missing">✗ {skill}</span>', unsafe_allow_html=True)
            else:
                st.markdown("*No missing skills — great!*")
        
        with s3:
            st.markdown("**⭐ Your Extra Skills**")
            if extra_skills:
                for skill in sorted(extra_skills):
                    st.markdown(f'<span class="skill-extra">+ {skill}</span>', unsafe_allow_html=True)
            else:
                st.markdown("*No extra skills detected*")
        
        st.divider()
        
        # Top JD Keywords
        st.markdown("### 🔑 Top Job Description Keywords")
        if jd_keywords:
            keyword_cols = st.columns(5)
            for i, (keyword, score_val) in enumerate(jd_keywords):
                with keyword_cols[i % 5]:
                    st.markdown(f"`{keyword}` — {score_val:.3f}")
        
        st.divider()
        
        # Strengths & Suggestions
        col_str, col_sug = st.columns(2)
        
        with col_str:
            st.markdown("### 💪 Strengths")
            if strengths:
                st.markdown('<div class="strength-box">', unsafe_allow_html=True)
                for s in strengths:
                    st.markdown(f"- {s}")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("Upload a more detailed resume for personalized strength analysis.")
        
        with col_sug:
            st.markdown("### 💡 Improvement Suggestions")
            if suggestions:
                st.markdown('<div class="suggestion-box">', unsafe_allow_html=True)
                for s in suggestions:
                    st.markdown(f"- {s}")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.success("Your resume looks well-matched! Keep it up.")
        
        # Experience & Education
        st.divider()
        e1, e2 = st.columns(2)
        with e1:
            st.markdown("### 📅 Experience")
            if resume_exp:
                st.markdown(f"**Your experience:** {resume_exp} years")
            else:
                st.markdown("*Could not detect years of experience in resume*")
            if jd_exp:
                st.markdown(f"**Required:** {jd_exp} years")
            else:
                st.markdown("*No specific experience requirement detected in JD*")
        
        with e2:
            st.markdown("### 🎓 Education")
            if resume_education:
                st.markdown(f"**Detected:** {', '.join(sorted(resume_education))}")
            else:
                st.markdown("*No education keywords detected in resume*")
            if education_match:
                st.markdown("✅ Education requirements appear to be met")
            else:
                st.markdown("⚠️ Check if your education matches the JD requirements")


if __name__ == "__main__":
    main()
