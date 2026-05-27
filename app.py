import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
import requests
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)


# Paths

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
BACKGROUND_FOLDER = os.path.join(BASE_DIR, "Background")

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# Background Route

@app.route('/background/<path:filename>')
def background_files(filename):
    return send_from_directory(BACKGROUND_FOLDER, filename)



# Load CNN Model

model = tf.keras.models.load_model("brain_tumor_best_model.keras")

IMG_SIZE = 224

classes = ['glioma', 'meningioma', 'notumor', 'pituitary']



# Prediction Function

def predict_img(img_path):

    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)

    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)

    predicted_class = classes[np.argmax(prediction)]
    confidence = float(np.max(prediction) * 100)

    return predicted_class, confidence



# AI Medical Report

def generate_ai_report(tumor, confidence, patient_name):

    prompt = f"""
You are a medical AI assistant.

Generate a medical report in EXACTLY this format.
Each heading must appear on a NEW LINE.

Patient Name: {patient_name}
Title: {tumor} Tumor Diagnosis Report
Diagnosis: {tumor}

Description:
Explain the tumor briefly using the confidence {confidence:.2f} %.

Recommendation:
Give medical recommendation for the patient.

Estimated Tumor Size:
Give estimated tumor size.

Risk Level:
Mention whether the risk is Low, Moderate, or High.

Rules:
- Do NOT repeat patient name again
- Do NOT include extra headings
- Do NOT include Patient Information
- Do NOT include Tumor Stage
- Keep the report clean and professional
"""

    try:

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False
            }
        )

        result = response.json()

        return result.get("response", "AI report could not be generated.")

    except:
        return "AI report service is not available. Make sure Ollama is running."



# Home Route

@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        if "image" not in request.files:
            return render_template("index.html")

        patient_name = request.form.get("patient_name")

        file = request.files["image"]

        if file.filename == "":
            return render_template("index.html")

        filename = secure_filename(file.filename)

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        file.save(filepath)

        # CNN Prediction
        result, confidence = predict_img(filepath)

        # AI Report
        report = generate_ai_report(result, confidence, patient_name)

        img_path = "uploads/" + filename

        # Current Date
        current_date = datetime.now().strftime("%d %B %Y")

        return render_template(
            "result.html",
            prediction=result,
            confidence=round(confidence, 2),
            img_path=img_path,
            report=report,
            patient_name=patient_name,
            current_date=current_date
        )

    return render_template("index.html")



# AI Chatbot

@app.route("/ask_ai", methods=["POST"])
def ask_ai():

    data = request.get_json()

    question = data.get("question", "")

    try:

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": question,
                "stream": False
            }
        )

        result = response.json()

        answer = result.get("response", "No response generated.")

    except:
        answer = "AI service is not available. Please make sure Ollama is running."

    return jsonify({"answer": answer})


# Run Flask

if __name__ == "__main__":
    app.run(debug=True)