from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from inference import predict_leaf_dishes  # Your function to run inference

app = Flask(__name__)
CORS(app)

# Routes to serve HTML pages
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ponjikkara')
def ponjikkara():
    return render_template('ponjikkara.html')

@app.route('/ponjikkara-output')
def ponjikkara_output():
    return render_template('ponjikkara-output.html')

@app.route('/dietchettan')
def dietchettan():
    return render_template('dietchettan.html')

@app.route('/dietchettan-output')
def dietchettan_output():
    return render_template('dietchettan-output.html')


# API endpoint to receive leaf image and return dish positions
@app.route('/upload-leaf', methods=['POST'])
def upload_leaf():
    if 'leaf' not in request.files:
        return jsonify({"error": "No leaf image uploaded"}), 400

    leaf_image = request.files['leaf']

    # Save uploaded image temporarily (optional)
    img_path = f"uploads/{leaf_image.filename}"
    leaf_image.save(img_path)

    # Call your inference code here — replace this with your actual inference function
    # For now, just dummy data
    try:
        # result = predict_leaf_dishes(img_path)  # Uncomment when inference.py ready
        # Dummy response (positions in pixels on leaf image)
        result = {
            "dishes": [
                {"name": "Parippu", "x": 100, "y": 50},
                {"name": "Sambar", "x": 150, "y": 100},
                {"name": "Avial", "x": 200, "y": 150},
                {"name": "Payasam", "x": 250, "y": 200}
            ]
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '_main_':
    app.run(debug=True)