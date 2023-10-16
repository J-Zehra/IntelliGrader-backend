import base64
import json
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import utils
import cv2

app = Flask(__name__)
CORS(app)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/process', methods=['POST'])
def process_image():
    if "image" not in request.files:
        return "No Image File Provided", 400

    image = request.files["image"]
    correct_answer = request.form.get("answer")
    correct_answer_indices = json.loads(correct_answer)

    # PROCESS IMAGE
    result = utils.process(image, 5, correct_answer_indices)

    if result is None:
        return jsonify({"error": "Not all circles are detected"}), 500

    _, buffer = cv2.imencode(".jpg", result["processed_image"])
    encoded_image = base64.b64encode(buffer).decode("utf-8")

    response_data = {
        "processed_image": encoded_image,
        "answer_indices": result["answer_indices"],
        "number_of_correct": result["number_of_correct"],
        "number_of_incorrect": result["number_of_incorrect"]
    }

    return jsonify(response_data), 200


if __name__ == '__main__':
    app.run(debug=True)
