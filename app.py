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

    for index, image in enumerate(data["images"], start=1):
        image_array = np.frombuffer(image, dtype=np.uint8)
        image_original = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        result = utils.process(image_original, data["parts"], data["answer"])
        result_status = result["status"]

        if result_status == "error":
            error_response = {
                "message": result["message"],
                "status": "failed",
                "image": image,
            }
            emit("progress", error_response)
            continue

        _, buffer = cv2.imencode(".webp", result["processed_image"])
        binary_image = buffer.tobytes()

        response_data = {
            "index": index,
            "status": "success",
            "processed_image": binary_image,
            "answer_indices": result["answer_indices"],
            "number_of_correct": result["number_of_correct"],
            "number_of_incorrect": result["number_of_incorrect"],
            "total_score": result["total_score"],
            "total_perfect_score": result["total_perfect_score"],
            "roll_number": result["roll_number"]
        }

        emit("progress", response_data)
        socketio.sleep(0.1)

    if len(data["images"]) > 1:
        emit("finished", {"status": "finished"})


if __name__ == '__main__':
    socketio.run(app, debug=True)
