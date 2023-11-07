import cv2
import numpy as np
import pytesseract


def process(image, number_of_choices, correct_answer_indices):
    image_original = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_COLOR)
    image_copy = image_original.copy()
    answer_indices = []
    number_of_correct = 0
    number_of_incorrect = 0

    # PREPROCESS IMAGE
    image_gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    image_blur = cv2.GaussianBlur(image_gray, (5, 5), 1)
    image_canny = cv2.Canny(image_blur, 10, 50)

    # FIND ALL CONTOURS
    contours, _ = cv2.findContours(image_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # FIND LARGEST AND SECOND LARGEST RECTANGLE
    bubble_section, roll_number_section = find_area_of_interest(contours, image_copy)

    # GET ROLL NUMBER
    roll_number = pytesseract.image_to_string(roll_number_section)

    bubble_section_gray = cv2.cvtColor(bubble_section, cv2.COLOR_BGR2GRAY)
    bubble_section_blur = cv2.GaussianBlur(bubble_section_gray, (5, 5), 1)

    # DETECT CIRCLES
    circles = cv2.HoughCircles(
        bubble_section_blur, cv2.HOUGH_GRADIENT, dp=1, minDist=25, param1=125, param2=20, minRadius=5, maxRadius=15
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        number_of_circles = len(circles)

        if number_of_circles is not (number_of_choices * len(correct_answer_indices)):
            print(f"Detected {number_of_circles} circles")
            return

        sorted_circles = sort_circles(circles, bubble_section)

        for i in range(0, len(circles), 5):
            question_circles = sorted_circles[i:i + 5]

            for index, (x, y, r) in enumerate(question_circles):
                roi_gray = bubble_section_gray[y - r:y + r, x - r:x + r]

                # Apply thresholding (you may need to fine-tune the threshold value)
                _, binary_roi = cv2.threshold(roi_gray, 10, 255, cv2.THRESH_BINARY)

                shading_percentage = get_shading_percentage(binary_roi)

                if shading_percentage <= 75:
                    answer_indices.append(index)
                    cv2.circle(bubble_section, (x, y), r, (0, 0, 255), 2)
                else:
                    cv2.circle(bubble_section, (x, y), r, (0, 255, 0), 2)

                cv2.putText(bubble_section, str(1 + i + index), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                            (225, 0, 0), 1)

    return {
        "processed_image": bubble_section,
        "original_image": image_original,
        "answer_indices": answer_indices,
        "number_of_correct": number_of_correct,
        "number_of_incorrect": number_of_incorrect,
        "roll_number": roll_number
    }


def find_area_of_interest(contours, image):
    largest_rect = None
    second_largest_rect = None
    max_area = 0
    second_max_area = 0

    # Iterate through detected contours
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the polygon has 4 vertices (a rectangle)
        if len(approx) == 4:
            # Calculate the area of the rectangle
            area = cv2.contourArea(contour)

            # Update the smaller and larger rectangles
            if area > max_area:
                second_max_area = max_area
                max_area = area
                second_largest_rect = largest_rect
                largest_rect = approx
            elif area > second_max_area:
                second_max_area = area
                second_largest_rect = approx

    # Crop the smaller and larger rectangles
    x, y, w, h = cv2.boundingRect(second_largest_rect)
    smaller_section = image[y:y + h, x:x + w]

    x, y, w, h = cv2.boundingRect(largest_rect)
    larger_section = image[y:y + h, x:x + w]

    return larger_section, smaller_section


def sort_circles(circles, cropped_bubble_image):
    first_column_circles = [circle for circle in circles if circle[0] < cropped_bubble_image.shape[1] / 2]
    second_column_circles = [circle for circle in circles if circle[0] >= cropped_bubble_image.shape[1] / 2]

    # Sort each list based on y-coordinate (row order)
    first_column_circles = sorted(first_column_circles, key=lambda circle: (circle[1], circle[0]))
    second_column_circles = sorted(second_column_circles, key=lambda circle: (circle[1], circle[0]))

    # Concatenate the sorted lists to get the final order
    sorted_circles = first_column_circles + second_column_circles

    return sorted_circles


def get_shading_percentage(roi):
    total_pixels = roi.size
    shaded_pixels = cv2.countNonZero(roi)
    shading_percentage = (shaded_pixels / total_pixels) * 100
    # print(shading_percentage)

    return shading_percentage
