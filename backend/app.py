from flask import Flask, request, jsonify, send_file
from PIL import Image
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return "Bolo Thararara Backend is running!"

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'leaf' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['leaf']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Placeholder: We'll do image magic here later
    return jsonify({'message': 'Leaf received', 'filename': file.filename})

if __name__ == '__main__':
    app.run(debug=True)
