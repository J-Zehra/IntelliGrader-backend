import base64
from flask import Flask
from flask_cors import CORS
from pytesseract import pytesseract

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


def decode_image(image):
    image_data = image.split(',')[1]
    image_bytes = b64decode(image_data)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    decoded_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    return decoded_image

@socketio.on('image')
def handle_image(data):
    template_marker = cv2.imread("marker.png", 0)
    template_marker_2 = cv2.imread("marker2.png", 0)

    image = decode_image(data)

    # PREPROCESS IMAGE
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    roll_number_section = utils.extract_section(image_gray, template_marker_2)
    bubble_section = utils.extract_section(image_gray, template_marker)

    if roll_number_section is not None and bubble_section is not None:
        print("success")
        _, roll_number_buffer = cv2.imencode(".jpg", roll_number_section)
        _, bubble_buffer = cv2.imencode(".jpg", bubble_section)
        encoded_roll_number_section = base64.b64encode(roll_number_buffer).decode("utf-8")
        encoded_bubble_section = base64.b64encode(bubble_buffer).decode("utf-8")
        data = {
            "rollNumberSection": encoded_roll_number_section,
            "bubbleSection": encoded_bubble_section
        }
        emit("request_test_data", data)


@socketio.on("single_grade")
def handle_process_images(data):
    roll_number_section = data["rollNumberSection"]
    bubble_section = data["bubbleSection"]
    answer = data["answer"]
    number_of_choices = data["numberOfChoices"]

    roll_number_section = decode_image(roll_number_section)
    bubble_section = decode_image(bubble_section)

    # GET ROLL
    roll_number = None

    try:
        roll_number = pytesseract.image_to_string(roll_number_section, config='--psm 11 digits')
        roll_number = int(roll_number)
    except Exception as e:
        print("Roll Number Not Detected")

    bubble_section_blur = cv2.GaussianBlur(bubble_section, (21, 21), 1)

    # DETECT CIRCLES
    circles = cv2.HoughCircles(
        bubble_section_blur, cv2.HOUGH_GRADIENT, dp=1, minDist=10, param1=80, param2=10, minRadius=5, maxRadius=8
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        number_of_circles = int(len(circles))
        detected_circles = int(
            sum(choice * len(answer) for choice in number_of_choices) / len(number_of_choices))
        print(f"{number_of_circles} circles")
        print(f"Detected {detected_circles} circles")

        if number_of_circles != detected_circles:
            return

        sorted_top_left, sorted_bottom_left, sorted_top_right, sorted_bottom_right = utils.sort_circles(circles,
                                                                                                  bubble_section,
                                                                                                  number_of_choices)

        try:
            choices_1 = number_of_choices[0]
        except IndexError:
            choices_1 = 1

        try:
            choices_2 = number_of_choices[1]
        except IndexError:
            choices_2 = 1

        try:
            choices_3 = number_of_choices[2]
        except IndexError:
            choices_3 = 1

        try:
            choices_4 = number_of_choices[3]
        except IndexError:
            choices_4 = 1

        part_1_answer_indices = utils.extract_answer_indices(sorted_top_left, choices_1, bubble_section)
        part_2_answer_indices = utils.extract_answer_indices(sorted_bottom_left, choices_2, bubble_section)
        part_3_answer_indices = utils.extract_answer_indices(sorted_top_right, choices_3, bubble_section)
        part_4_answer_indices = utils.extract_answer_indices(sorted_bottom_right, choices_4, bubble_section)

        answer_indices = part_1_answer_indices + part_2_answer_indices + part_3_answer_indices + part_4_answer_indices

        number_of_correct, number_of_incorrect = utils.check(answer_indices, answer)

        _, buffer = cv2.imencode(".jpg", bubble_section)
        encoded_image = base64.b64encode(buffer).decode("utf-8")

        data = {
            "processed_image": encoded_image,
            "answer_indices": answer_indices,
            "number_of_correct": number_of_correct,
            "number_of_incorrect": number_of_incorrect,
            "roll_number": roll_number
        }

        emit("single_grade_data", data)
        print(number_of_correct, number_of_incorrect)


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



if __name__ == '__main__':
    socketio.run(app, debug=True)
