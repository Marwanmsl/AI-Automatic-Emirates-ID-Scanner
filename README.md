# 🤖 AI Automatic Emirates ID Scanner

An AI-powered desktop application for automatically detecting and extracting information from Emirates ID cards using a webcam, OpenCV, PyQt5, and Ollama vision models.

https://github.com/user-attachments/assets/3be34ff4-c098-471e-bc15-04f0afd97a26

## 🚀 Features

* 📷 Real-time webcam-based Emirates ID detection
* 🎯 Automatic ID positioning guide
* ⚡ Automatic image capture when the ID remains stable
* 🧠 AI-powered OCR and information extraction using Ollama
* 📋 Structured JSON extraction
* 👤 Extracts:

  * Full Name
  * ID Number
  * Nationality
  * Date of Birth
  * Issuing Date
  * Expiry Date
* 🖥️ Professional PyQt5 graphical interface
* 📐 Resizable camera and information panels
* 💾 Save extracted information as a JSON file
* 🔄 Automatic scanner reset for processing the next ID
* 📴 Designed for local/offline AI processing when a locally available Ollama vision model is used

## 🛠️ Technologies

* Python
* OpenCV
* PyQt5
* Ollama
* JSON
* Regular Expressions
* Computer Vision
* Vision Language Model (VLM)

## 🔄 How It Works

1. The webcam continuously captures video.
2. The application analyzes the region inside the ID guide.
3. Computer vision techniques detect edges, image variance, and contours.
4. When an ID is detected and remains stable, the application automatically captures an image.
5. The captured image is sent to an Ollama vision model.
6. The AI extracts the required Emirates ID information.
7. The response is converted into structured JSON.
8. The extracted information is displayed in the application.
9. The user can save the extracted data as a JSON file.

## 📦 Main Components

### Automatic Detection

OpenCV analyzes the camera region using Canny edge detection, image variance, and contour analysis to determine whether an ID is present.

### AI Information Extraction

The application sends the captured ID image to an Ollama vision model with a structured extraction prompt and requests JSON output.

### JSON Processing

AI responses are cleaned and parsed so that structured ID information can be displayed and saved.

### Data Export

Extracted information can be saved locally as a JSON file.

## 💻 Installation

Install the required Python packages:

```bash
pip install opencv-python PyQt5 ollama
```

Make sure Ollama is installed and the required vision model is available locally.

Update the model configuration if required:

```python
OLLAMA_MODEL = "gemma4:31b-cloud"
```

Then run:

```bash
python main.py
```

## ⚠️ Privacy & Security

This project processes sensitive identity-document information. Use it only with appropriate authorization and follow applicable privacy and data-protection requirements.

Do not upload real Emirates ID images, extracted identity information, or other sensitive personal data to a public GitHub repository.

## 📌 Project Purpose

This project demonstrates the integration of:

**Computer Vision + Vision Language Models + OCR-style Information Extraction + Desktop GUI + Automated Document Capture**

It can serve as a foundation for intelligent document-scanning and identity-document processing applications.
