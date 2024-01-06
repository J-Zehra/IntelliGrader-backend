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


def decode_encoded_image(image):
    # Decode base64 string into binary data
    binary_data = base64.b64decode(image)

    # Convert binary data into NumPy array
    np_array = np.frombuffer(binary_data, dtype=np.uint8)

    # Decode the image using OpenCV
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    return image


# Initialize state variables
stored_roll_number_section = None
stored_bubble_section = None


@socketio.on('image')
def handle_image(data):
    global stored_roll_number_section, stored_bubble_section

    response_data = None
    template_marker = cv2.imread("marker.png", 0)
    template_marker_2 = cv2.imread("marker2.png", 0)

    image = decode_image(data)

    # Only detect sections if not already stored
    if stored_roll_number_section is None:
        roll_number_section, _ = utils.extract_section(image, template_marker_2)
        if roll_number_section is not None:
            stored_roll_number_section = roll_number_section

    if stored_bubble_section is None:
        bubble_section, _ = utils.extract_section(image, template_marker)
        if bubble_section is not None:
            stored_bubble_section = bubble_section

    if stored_roll_number_section is not None and stored_bubble_section is not None:
        print("success")

        _, roll_number_buffer = cv2.imencode(".webp", stored_roll_number_section)
        _, bubble_buffer = cv2.imencode(".webp", stored_bubble_section)
        encoded_roll_number_section = base64.b64encode(roll_number_buffer).decode("utf-8")
        encoded_bubble_section = base64.b64encode(bubble_buffer).decode("utf-8")

        response_data = {
            "rollNumberSection": encoded_roll_number_section,
            "bubbleSection": encoded_bubble_section,
            "status": "success"
        }

        stored_roll_number_section = None
        stored_bubble_section = None

        return response_data

    callback = {"status": "failed"}
    return callback


@socketio.on("single_grade")
def handle_process_images(data):
    roll_number_section = data["rollNumberSection"]
    bubble_section = data["bubbleSection"]
    answer = data["answer"]
    number_of_choices = data["numberOfChoices"]

    print("Single Grade")

    roll_number_section = decode_encoded_image(roll_number_section)
    bubble_section = decode_encoded_image(bubble_section)

    # GET ROLL
    roll_number = pytesseract.image_to_string(roll_number_section, config='--psm 11 digits')

    try:
        roll_number = int(roll_number)
    except ValueError:
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

        _, buffer = cv2.imencode(".webp", bubble_section)
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
    parts = data["parts"]

    response_data = []
    for image in images:
        image_data = image.split(',')[1]
        image_bytes = b64decode(image_data)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image_original = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        result = utils.process(image_original, parts, answer)

        if result["status"] == "error":
            print("error")
            error_response = {
                "message": result["message"],
                "status": result["status"],
                "image": image,
            }
            response_data.append(error_response)
            continue

        _, buffer = cv2.imencode(".webp", result["processed_image"])
        encoded_image = base64.b64encode(buffer).decode("utf-8")

        data = {
            "status": "success",
            "processed_image": encoded_image,
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
