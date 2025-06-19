from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import openai
from dotenv import load_dotenv
import random
from PIL import Image

# Load environment variables
load_dotenv()

# Initialize Flask App
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "supersecretkey")

# Configure Database
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///database.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# Load AI Model
model_path = r"D:\Coding\fresh_Rot_fruits\model\fruit_classification_model2.keras"
model = tf.keras.models.load_model(model_path)

# Ensure upload folder exists
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Class labels as per your model
class_names = ['none','freshapples', 'freshbanana', 'rottenapples', 'rottenbanana']

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(100), nullable=False)
    prediction = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)

# Create database tables
with app.app_context():
    db.create_all()

# Initialize OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to predict the class of the image
def predict(model, img):
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, confidence

# Function to check if the image resembles an apple or banana
def is_valid_image(img):
    """
    Check if the image resembles an apple or banana.
    This is a simple check and can be improved with a more sophisticated method.
    """
    # Convert image to grayscale
    gray_img = img.convert("L")
    # Calculate average pixel intensity
    avg_intensity = np.mean(gray_img)
    # Simple thresholding (adjust as needed)
    if avg_intensity < 50 or avg_intensity > 200:
        return False
    return True

# Predefined QA responses for chatbot
qa_pairs = {
    "hello": "Hello, I am your Fruity!",
    "hi": "Hi, I am your Fruity!",
    "what is your name": "My name is Fruity, your intelligent assistant for this project!",
    "what is this project about": "This project is on machine learning that can determine whether an apple or banana is fresh or rotten.",
    "how does the model work": "This ML model uses a Convolutional Neural Network (CNN) to analyze images and classify them as fresh or rotten.",
    "what is cnn": "CNN, or Convolutional Neural Network, is a deep learning algorithm designed to process image data and extract important features for classification tasks.",
    "what is tensorflow": "TensorFlow is an open-source machine learning framework developed by Google for building and training neural networks efficiently.",
    "what is keras": "Keras is a high-level deep learning API built on TensorFlow that makes it easier to design and train neural networks.",
    "what is machine learning": "Machine Learning is a branch of AI that enables computers to learn from data and make predictions without being explicitly programmed.",
    "what is deep learning": "Deep Learning is a subset of Machine Learning that uses neural networks with multiple layers to learn complex patterns in data.",
    "what is ai": "AI, or Artificial Intelligence, is a field of computer science focused on creating intelligent systems that can mimic human cognitive functions.",
    "what is a chatbot": "A chatbot is an AI-powered program designed to simulate conversations with users and provide automated responses based on inputs.",
    "is there a login system": "Yes, our system includes a separate login and registration system for users.",
    "can i see previous history": "Yes, you can download and view your previous history of queries and results.",
    "can i navigate between pages": "Yes, our web app allows you to easily move back and forth between different pages.",
    "who developed this project": "This project was developed by a team of AI enthusiasts dedicated to building intelligent solutions.",
    "why use deep learning for this project": "Deep learning is ideal for this project because it can analyze images with high accuracy, extracting important features for classification.",
    "how does cnn help in image classification": "CNN (Convolutional Neural Networks) helps by detecting patterns like edges, textures, and shapes, making it effective for image-based tasks.",
}

# List of random fallback responses
random_responses = [
    "That's an interesting question!",
    "I'm still learning. Can you ask that in a different way?",
    "I don't have an exact answer for that, but I'm here to help!",
    "Hmm... I'm not sure. Let's explore that together!",
    "That's a great question! What do you think?",
    "I'm just a chatbot, but I'm happy to chat!",
    "Let me get back to you on that!",
]

# Chatbot Route
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "").strip().lower()
    bot_reply = qa_pairs.get(user_message, random.choice(random_responses))
    return jsonify({"response": bot_reply})

# Landing Page
@app.route("/")
def landing():
    return render_template("landing.html")

# Login Route
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            session["user_id"] = user.id
            flash("Login successful!", "success")
            return redirect(url_for("home"))
        else:
            flash("Invalid username or password.", "error")

    return render_template("login.html")

# Registration Route
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        hashed_password = generate_password_hash(password, method="pbkdf2:sha256")

        if User.query.filter_by(username=username).first():
            flash("Username already exists!", "error")
            return redirect(url_for("register"))

        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash("Registration successful! Please login.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")

# Home Route (Upload & Predict)
@app.route("/home", methods=["GET", "POST"])
def home():
    if "user_id" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            flash("No file selected. Please upload an image.", "error")
            return redirect(url_for("home"))

        # Save the file
        filename = file.filename
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        try:
            # Process the image
            img = image.load_img(filepath, target_size=(224, 224))

            # Check if the image is valid (resembles an apple or banana)
            if not is_valid_image(img):
                flash("Invalid image. Please upload an image of an apple or banana.", "error")
                return redirect(url_for("home"))

            # Predict the class
            predicted_class, confidence = predict(model, img)

            # Store prediction in the database
            new_prediction = Prediction(filename=filename, prediction=predicted_class, confidence=confidence)
            db.session.add(new_prediction)
            db.session.commit()

            flash(f"Prediction: {predicted_class} (Confidence: {confidence}%)", "success")
            return redirect(url_for("home"))
        except Exception as e:
            flash(f"Error processing image: {str(e)}", "error")
            return redirect(url_for("home"))

    return render_template("home.html")

# Prediction Page (Show Image & Prediction)
@app.route("/prediction/<prediction_id>")
def prediction(prediction_id):
    if "user_id" not in session:
        return redirect(url_for("login"))

    prediction = Prediction.query.get_or_404(prediction_id)
    img_path = os.path.join(app.config["UPLOAD_FOLDER"], prediction.filename)
    return render_template("prediction.html", prediction=prediction, img_path=img_path)

# Prediction History
@app.route("/history")
def history():
    if "user_id" not in session:
        return redirect(url_for("login"))

    predictions = Prediction.query.order_by(Prediction.id.desc()).all()
    return render_template("history.html", predictions=predictions)

# Clear History
@app.route("/clear_history", methods=["POST"])
def clear_history():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    try:
        # Delete all predictions from the database
        Prediction.query.delete()
        db.session.commit()

        # Delete uploaded files
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        flash("History cleared successfully.", "success")
        return jsonify({"message": "History cleared successfully"}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

# Logout Route
@app.route("/logout", methods=["POST"])
def logout():
    session.pop("user_id", None)  # Clear the session
    flash("Logged out successfully!", "success")
    return redirect(url_for("landing"))

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)