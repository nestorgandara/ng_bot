# Nestor’s Bot 🤖

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)  
[![Streamlit](https://img.shields.io/badge/Streamlit-App-green?logo=streamlit)](https://streamlit.io/)  
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-orange)](https://github.com/facebookresearch/faiss)  

A sleek, **AI-powered document assistant** built with Streamlit, FAISS, and AWS Bedrock. Upload documents (PDF, TXT, DOCX), ask questions, and get context-aware answers with optional summaries.

![Bot Logo](bot_app.png)

---

## 🚀 Features

- **Document Upload**: Supports PDF, TXT, DOCX.  
- **Vector Search**: Uses FAISS to embed and retrieve relevant content.  
- **AI Q&A**: Contextual answers using Claude 3 via AWS Bedrock.  
- **Summary Mode**: Condensed 3-bullet summaries of answers.  
- **Interactive UI**: Modern hero design, floating logo, glow effect, and particle background for a visually engaging experience.  

---

## 📦 Requirements

- Python 3.9+  
- `streamlit`  
- `boto3`  
- `python-dotenv`  
- `faiss-cpu`  
- `PyPDF2`  
- `docx2txt`  
