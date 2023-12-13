import base64
import json
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import utils
import cv2
import numpy as np
from flask_socketio import SocketIO, emit
from base64 import b64decode

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")


@socketio.on("connect")
def handle_connect():
    emit("connect_emit", "Successfully Connected")
    print("Client Connected")


@socketio.on("grade")
def handle_process_images(data):
    images = data["images"]
    answer = data["answer"]
    number_of_choices = data["numberOfChoices"]

    print(images)

    response_data = []
    for image in images:
        image_data = image.split(',')[1]
        image_bytes = b64decode(image_data)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image_original = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        result = utils.process(image_original, number_of_choices, answer)

        if result is None:
            data = {"status": "invalid"}
            response_data.append(data)
            return

        _, buffer = cv2.imencode(".jpg", result["processed_image"])
        encoded_image = base64.b64encode(buffer).decode("utf-8")

        data = {
            "status": "success",
            "processed_image": encoded_image,
            "answer_indices": result["answer_indices"],
            "number_of_correct": result["number_of_correct"],
            "number_of_incorrect": result["number_of_incorrect"],
            "roll_number": result["roll_number"]
        }

        response_data.append(data)

    emit("grade_result", response_data)


# @app.route('/')
# def hello_world():
#     return 'Hello World!'


# @app.route('/process', methods=['POST'])
# def process_image():
#     if 'images' not in request.files:
#         return 'No images part in the request', 400
#
#     images = request.files.getlist('images')
#     correct_answer = request.form.get("answer")
#     number_of_choices = request.form.get("numberOfChoices")
#     correct_answer_indices = json.loads(correct_answer)
#     number_of_choices_array = json.loads(number_of_choices)
#
#     response_data = []
#     for image in images:
#         image_original = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
#         result = utils.process(image_original, number_of_choices_array, correct_answer_indices)
#
#         if result is None:
#             data = {"status": "invalid"}
#             response_data.append(data)
#             return
#
#         _, buffer = cv2.imencode(".jpg", result["processed_image"])
#         encoded_image = base64.b64encode(buffer).decode("utf-8")
#
#         data = {
#             "status": "success",
#             "processed_image": encoded_image,
#             "answer_indices": result["answer_indices"],
#             "number_of_correct": result["number_of_correct"],
#             "number_of_incorrect": result["number_of_incorrect"],
#             "roll_number": result["roll_number"]
#         }
#
#         response_data.append(data)
#
#     return jsonify(response_data), 200


if __name__ == '__main__':
    socketio.run(app, debug=True)
