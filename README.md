<p align="center">
  <img src="docs/banner.png" alt="PicIntel Banner" width="100%">
</p>

# 🛰️ PicIntel

**AI-powered Image Intelligence for the Digital Age**

---

### 🧩 Overview

Images can be manipulated, misused, or taken out of context — spreading misinformation and deepfakes.

That’s where **PicIntel** comes in:  
An **AI-driven OSINT (Open Source Intelligence)** platform that verifies image authenticity, traces its origin, and uncovers hidden metadata — all in one click.

---

### 🔍 Features

- **Authenticity Check:** Detects manipulations using deepfake & ELA analysis.  
- **Metadata Intelligence:** Extracts EXIF data, GPS info, timestamps, and OCR text.  
- **Reverse Image Search:** Finds similar images across the web.  
- **Automated Reporting:** Generates a PDF report with confidence scores.

---

### ⚙️ Tech Stack

`Flask` • `OpenCV` • `Pillow` • `pytesseract` • `SerpApi` • `SQLite3` • `Hugging Face models`

---

### 🧱 Setup

```bash
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate   # for Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
