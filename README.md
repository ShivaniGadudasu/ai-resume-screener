# 📄 AI Resume Screener (Streamlit App)

An AI-powered Resume Screening App built with Python and Streamlit that helps HR professionals automatically rank resumes based on similarity to a job description.

## 🚀 Features
- Upload job description (`.txt`, `.docx`, `.pdf`)
- Upload resumes in a `.zip` file
- Automatically extract and read resumes
- Ranks resumes using TF-IDF and cosine similarity
- Download results as CSV

## 🛠️ Installation
```bash
git clone https://github.com/ShivaniGadudasu/ai-resume-screener.git
cd ai-resume-screener
pip install -r requirements.txt
streamlit run app.py
