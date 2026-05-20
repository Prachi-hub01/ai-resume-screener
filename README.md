# 📄 AI Resume Screener — Smart Resume-Job Matching with NLP

> Intelligent resume analysis tool that scores how well your resume matches a job description using NLP and Sentence Transformers.

---

## 📌 Overview

AI Resume Screener is a Streamlit web application that helps job seekers optimize their resumes for specific job descriptions. It uses a combination of **Semantic Similarity (Sentence-BERT)**, **TF-IDF keyword matching**, and **skill extraction** to provide an overall match score along with actionable improvement suggestions.

---

## ✨ Key Features

- 🤖 **Semantic Analysis** — Uses `all-MiniLM-L6-v2` Sentence-BERT model to understand contextual meaning beyond keyword matching
- 📊 **TF-IDF Keyword Matching** — Identifies important terms and measures keyword overlap
- 🎯 **Skill Extraction** — Automatically detects 200+ technical and soft skills from both resume and job description
- 💡 **Actionable Suggestions** — Provides personalized improvement tips based on gaps found
- 📄 **PDF Upload Support** — Upload resume as PDF or paste text directly
- 📅 **Experience Detection** — Extracts and compares years of experience
- 🎓 **Education Matching** — Checks education qualification alignment

---

## 🖥️ App Screenshots

### Home Page
The main interface with sidebar info, resume upload (PDF or text), and job description input.

<p align="center">
  <img src="screenshots/01_home_page.png" alt="Home Page" width="800"/>
</p>

### Upload Resume & Job Description
Upload your resume as PDF and paste the job description to analyze.

<p align="center">
  <img src="screenshots/02_upload_resume.png" alt="Upload Resume and Job Description" width="800"/>
</p>

### Analysis Results & Skills Breakdown
Overall match score with semantic, keyword, and skills match percentages. Matched, missing, and extra skills are highlighted.

<p align="center">
  <img src="screenshots/03_analysis_results.png" alt="Analysis Results and Skills" width="800"/>
</p>

### Top Keywords & Improvement Suggestions
Top job description keywords with TF-IDF scores, strengths, and actionable improvement suggestions.

<p align="center">
  <img src="screenshots/04_keywords_suggestions.png" alt="Keywords and Suggestions" width="800"/>
</p>

### Experience & Education Matching
Experience years detection and education qualification alignment check.

<p align="center">
  <img src="screenshots/05_experience_education.png" alt="Experience and Education" width="800"/>
</p>

---

## 🧠 How It Works

The app computes a weighted **Overall Match Score** using:

| Component | Weight | Method |
|---|---|---|
| Semantic Similarity | 35% | Sentence-BERT (all-MiniLM-L6-v2) |
| Keyword Similarity | 25% | TF-IDF + Cosine Similarity |
| Skill Match | 35% | Custom skill extraction from 200+ skills database |
| Education Bonus | 5% | Education keyword matching |

### Scoring Guide

| Score | Rating |
|---|---|
| 85–100% | 🟢 Excellent Match |
| 70–84% | 🔵 Strong Match |
| 55–69% | 🟡 Good Match |
| 40–54% | 🟠 Fair Match |
| 0–39% | 🔴 Weak Match |

---

## 🛠️ Tech Stack

- **Python** — Core language
- **Streamlit** — Web app framework
- **Sentence-Transformers** — Semantic similarity (all-MiniLM-L6-v2)
- **Scikit-learn** — TF-IDF vectorization and cosine similarity
- **PyMuPDF (fitz)** — PDF text extraction
- **NumPy** — Numerical operations

---

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/Prachi-hub01/ai-resume-screener.git
   cd ai-resume-screener
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   streamlit run app.py
   ```

4. Open `http://localhost:8501` in your browser

---

## 📁 Project Structure

```
ai-resume-screener/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── screenshots/            # App UI screenshots
└── README.md               # This file
```

---

## 🔮 Future Improvements

- Add support for DOCX resume uploads
- Integrate with OpenAI/Claude API for AI-powered rewriting suggestions
- Add ATS (Applicant Tracking System) compatibility scoring
- Batch processing for multiple resumes
- Export analysis report as PDF

---

## 👩‍💻 Author

**Prachi Arvind Dutt** — MSc Artificial Intelligence, University of East London

---

## 📄 License

This project is open source and available for educational purposes.
