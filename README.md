# 📝 AI Text Summarizer

An AI-powered Text Summarization application built using **Streamlit**, **Hugging Face Transformers**, and **Python**. The application generates concise and meaningful summaries from multiple input sources, including text, documents, webpages, YouTube videos, and uploaded videos.

---

## 📌 Project Overview

This project provides an intelligent and user-friendly interface for summarizing large amounts of text. It uses state-of-the-art Natural Language Processing (NLP) models to extract key information and present it in a shorter, readable format.

The application supports multiple input formats, making it useful for students, researchers, professionals, and anyone who needs to understand lengthy content quickly.

---

## ✨ Features

- 📄 Summarize pasted text
- 🌐 Summarize webpage URLs
- 🎥 Summarize YouTube videos using transcripts
- 📑 Upload and summarize PDF files
- 📄 Upload and summarize DOCX files
- 📃 Upload and summarize TXT files
- 🎬 Upload narrated videos and generate summaries
- 🎚 Adjustable summary length (Short, Medium, Long)
- ⚡ Fast transcript generation
- 🎨 Modern and responsive Streamlit interface

---

## 🛠 Technologies Used

### Programming Language
- Python

### Framework
- Streamlit

### Machine Learning / NLP
- Hugging Face Transformers
- DistilBART CNN Model
- Whisper Tiny (Automatic Speech Recognition)

### Libraries
- PyMuPDF
- python-docx
- MoviePy
- imageio-ffmpeg
- youtube-transcript-api
- NumPy
- Pandas
- Torch

---

## 🧠 AI Models Used

### Text Summarization
- **DistilBART CNN 12-6**
- Model:
```
sshleifer/distilbart-cnn-12-6
```

### Speech Recognition
- **Whisper Tiny English**
- Model:
```
openai/whisper-tiny.en
```

---

## 📂 Supported Input Types

- Paste Text
- Webpage URL
- YouTube URL
- PDF
- DOCX
- TXT
- MP4
- MOV
- AVI
- MKV

---

## 📸 Application Workflow

```
User Input
      │
      ▼
Text / PDF / DOCX / TXT / URL / Video
      │
      ▼
Text Extraction / Speech Recognition
      │
      ▼
Text Cleaning
      │
      ▼
BART Summarization
      │
      ▼
Generated Summary
```

---

## 🚀 Installation

### Clone the repository

```bash
git clone https://github.com/Deekshitha16-ux/AI-Text-Summarizer.git
```

### Navigate to the project

```bash
cd AI-Text-Summarizer
```

### Create a virtual environment

```bash
python -m venv .venv
```

### Activate virtual environment

Windows

```bash
.venv\Scripts\activate
```

Linux / macOS

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

The application will open in your default web browser.

---

## 📁 Project Structure

```
AI-Text-Summarizer/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── test_videos/
```

---

## 📷 Screenshots

Add screenshots of your application here.

Example:

- Home Page
- Text Summarization
- PDF Upload
- URL Summarization
- Video Summarization
- Generated Summary

---

## 🎯 Applications

- Students
- Researchers
- Content Creators
- Journalists
- Professionals
- Online Learning
- Document Analysis

---

## 🔮 Future Enhancements

- Support multiple languages
- OCR for scanned PDFs
- Audio file summarization
- Cloud deployment
- User authentication
- Download summary as PDF
- Summary history

---

## 👩‍💻 Author

**Deekshitha R**

Computer Science & Engineering Student

GitHub:
https://github.com/Deekshitha16-ux

---

## 📄 License

This project is developed for educational and learning purposes.

---

## ⭐ If you found this project useful

Please consider giving this repository a ⭐ on GitHub.
