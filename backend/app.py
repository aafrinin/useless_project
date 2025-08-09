import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from inference import predict_leaf_dishes

app = Flask(__name__)
CORS(app)

# Ensure uploads folder exists
os.makedirs("uploads", exist_ok=True)

# --- HTML Routes ---
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ponjikkara")
def ponjikkara():
    return render_template("ponjikkara.html")

@app.route("/ponjikkara-output")
def ponjikkara_output():
    return render_template("ponjikkara-output.html")

@app.route("/dietchettan")
def dietchettan():
    return render_template("dietchettan.html")

@app.route("/dietchettan-output")
def dietchettan_output():
    return render_template("dietchettan-output.html")

@app.route('/')
def index():
    return render_template('index.html')

# --- API Route for Image Upload ---
@app.route("/upload-leaf", methods=["POST"])
def upload_leaf():
    if "leaf" not in request.files:
        return jsonify({"error": "No leaf image uploaded"}), 400

    leaf_image = request.files["leaf"]
    img_path = os.path.join("uploads", leaf_image.filename)
    leaf_image.save(img_path)

    try:
        result = predict_leaf_dishes(img_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
