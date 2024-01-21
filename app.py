import base64

from flask import Flask
from flask_cors import CORS

import utils
import cv2
import numpy as np
from flask_socketio import SocketIO, emit

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

@socketio.on("grade")
def handle_grade(data):
    print("HERE")
    images = data["images"]
    answer = data["answer"]
    parts = data["parts"]

    response_data = []
    for index, image in enumerate(images):
        print(index)
        image_array = np.frombuffer(image, dtype=np.uint8)
        image_original = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        result = utils.process(image_original, parts, answer)

        if result["status"] == "error":
            print("error")
            error_response = {
                "message": result["message"],
                "status": "failed",
                "image": image,
            }
            response_data.append(error_response)
            continue

        # socketio.emit("progress", {"index": index + 1})

        _, buffer = cv2.imencode(".webp", result["processed_image"])
        binary_image = buffer.tobytes()

        data = {
            "status": "success",
            "processed_image": binary_image,
            "answer_indices": result["answer_indices"],
            "number_of_correct": result["number_of_correct"],
            "number_of_incorrect": result["number_of_incorrect"],
            "total_score": result["total_score"],
            "total_perfect_score": result["total_perfect_score"],
            "roll_number": result["roll_number"]
        }

        response_data.append(data)

    emit("grade_result", response_data)


if __name__ == '__main__':
    socketio.run(app, debug=True)