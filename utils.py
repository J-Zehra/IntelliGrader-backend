import cv2
import numpy as np
import pytesseract


def process(image, number_of_choices, correct_answer_indices):
    template_marker = cv2.imread("marker.png", 0)
    image_copy = image.copy()
    answer_indices = []

    # PREPROCESS IMAGE
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_blur = cv2.GaussianBlur(image_gray, (5, 5), 0)
    image_canny = cv2.Canny(image_blur, 20, 75)

    # FIND ALL CONTOURS
    contours, _ = cv2.findContours(image_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # FIND LARGEST AND SECOND LARGEST RECTANGLE
    roll_number_section = find_roll_number_region(contours, image_copy)

    # GET ROLL
    roll_number = pytesseract.image_to_string(roll_number_section, config='--psm 11 digits')
    roll_number = int(roll_number)

    bubble_section = extract_bubble_section(image_gray, template_marker)
    bubble_section_blur = cv2.GaussianBlur(bubble_section, (21, 21), 1)

    # DETECT CIRCLES
    circles = cv2.HoughCircles(
        bubble_section_blur, cv2.HOUGH_GRADIENT, dp=1, minDist=10, param1=125, param2=12, minRadius=6, maxRadius=8
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        number_of_circles = len(circles)

        # if number_of_circles is not (sum(choice * len(correct_answer_indices) for choice in number_of_choices) / len(number_of_choices)):
        #     print(f"Detected {number_of_circles} circles")
        #     return

        sorted_top_left, sorted_bottom_left, sorted_top_right, sorted_bottom_right = sort_circles(circles,
                                                                                                  bubble_section,
                                                                                                  number_of_choices)

        part_1_answer_indices = extract_answer_indices(sorted_top_left, number_of_choices[0], bubble_section,
                                                       bubble_section)
        part_2_answer_indices = extract_answer_indices(sorted_bottom_left, number_of_choices[1], bubble_section,
                                                       bubble_section)
        part_3_answer_indices = extract_answer_indices(sorted_top_right, number_of_choices[2], bubble_section,
                                                       bubble_section)
        part_4_answer_indices = extract_answer_indices(sorted_bottom_right, number_of_choices[3], bubble_section,
                                                       bubble_section)

        answer_indices = part_1_answer_indices + part_2_answer_indices + part_3_answer_indices + part_4_answer_indices

    number_of_correct, number_of_incorrect = check(answer_indices, correct_answer_indices)

    return {
        "processed_image": bubble_section,
        "original_image": image,
        "answer_indices": answer_indices,
        "number_of_correct": number_of_correct,
        "number_of_incorrect": number_of_incorrect,
        "roll_number": roll_number
    }


def extract_bubble_section(sample_image, template_marker, scale_range=(0.5, 2.0), scale_step=0.1):
    bubble_section = None

    # Generate a range of scales
    scales = np.arange(scale_range[0], scale_range[1] + scale_step, scale_step)

    for scale in scales:
        # Resize the template at the current scale
        resized_template = cv2.resize(template_marker, None, fx=scale, fy=scale)

        # Match the resized template with the sample image
        result = cv2.matchTemplate(sample_image, resized_template, cv2.TM_CCOEFF_NORMED)

        # Set a threshold to consider a match
        threshold = 0.8
        loc = np.where(result >= threshold)

        # Get the coordinates of all the detected matches
        detected_positions = []
        for pt in zip(*loc[::-1]):
            detected_positions.append(pt)

        # If at least one match is found
        if detected_positions:
            # Convert to NumPy array for easier calculations
            detected_positions = np.array(detected_positions)

            # Compute the bounding box around all detected matches
            min_x, min_y = np.min(detected_positions, axis=0)
            max_x, max_y = np.max(detected_positions, axis=0)

            # Extract the region defined by the bounding box
            bubble_section = sample_image[min_y:max_y, min_x:max_x]

    return bubble_section


def find_roll_number_region(contours, image):
    smallest_rect = None
    target_ratio = 1 / 3
    min_area = float('inf')

    # Iterate through detected contours
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the polygon has 4 vertices (a rectangle)
        if len(approx) == 4:
            # Calculate the area of the rectangle
            area = cv2.contourArea(contour)

            # Calculate the bounding box for the rectangle
            x, y, w, h = cv2.boundingRect(approx)

            # Calculate the height-to-width ratio
            ratio = h / w

            # Update the smallest rectangle based on area and ratio
            if area < min_area and abs(ratio - target_ratio) < 0.1:  # Adjust the tolerance as needed
                min_area = area
                smallest_rect = approx

    # Crop the smallest rectangle
    x, y, w, h = cv2.boundingRect(smallest_rect)
    roll_number_section = image[y:y + h, x:x + w]

    return roll_number_section


def extract_answer_indices(sorted_circles, number_of_choices, bubble_section_gray, bubble_section):
    answer_indices = []

    for i in range(0, len(sorted_circles), number_of_choices):
        question_circles = sorted_circles[i:i + number_of_choices]
        shaded_index = -1
        shading_count = 0

        for index, (x, y, r) in enumerate(question_circles):
            roi_gray = bubble_section_gray[y - r:y + r, x - r:x + r]
            roi_blur = cv2.GaussianBlur(roi_gray, (21, 21), 1)
            roi_thresh = cv2.adaptiveThreshold(roi_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 41, 40)

            average_intensity = cv2.mean(roi_thresh)[0]
            shading_percentage = (average_intensity / 255) * 100

            print(shading_percentage)

            if shading_percentage > 50:
                shaded_index = index
                shading_count += 1
                cv2.circle(bubble_section, (x, y), r, (0, 0, 255), 2)
            else:
                cv2.circle(bubble_section, (x, y), r, (0, 255, 0), 2)

            cv2.putText(bubble_section, str(1 + index), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        (225, 0, 0), 1)

        if shading_count > 1:
            answer_indices.append(-2)
        elif shading_count == 1:
            answer_indices.append(shaded_index)
        else:
            answer_indices.append(-1)

    return answer_indices


def sort_circles(circles, cropped_bubble_image, number_of_choices):
    choices_1 = number_of_choices[0]
    choices_2 = number_of_choices[1]
    choices_3 = number_of_choices[2]
    choices_4 = number_of_choices[3]

    # Calculate the center of the image
    center_x = cropped_bubble_image.shape[1] // 2
    center_y = cropped_bubble_image.shape[0] // 2

    # Separate circles into four quadrants based on x and y coordinates
    top_left_circles = [circle for circle in circles if circle[0] < center_x and circle[1] < center_y]
    bottom_left_circles = [circle for circle in circles if circle[0] < center_x and circle[1] >= center_y]
    top_right_circles = [circle for circle in circles if circle[0] >= center_x and circle[1] < center_y]
    bottom_right_circles = [circle for circle in circles if circle[0] >= center_x and circle[1] >= center_y]

    # Sort circles within each quadrant based on y-coordinate (row order) and then x-coordinate (column order)
    top_left_circles = sorted(top_left_circles, key=lambda circle: (circle[1], circle[0]))
    bottom_left_circles = sorted(bottom_left_circles, key=lambda circle: (circle[1], circle[0]))
    top_right_circles = sorted(top_right_circles, key=lambda circle: (circle[1], circle[0]))
    bottom_right_circles = sorted(bottom_right_circles, key=lambda circle: (circle[1], circle[0]))

    # Sort the circles for each quadrant based on your specific sorting function (sort)
    sorted_top_left = sort(choices_1, top_left_circles)
    sorted_bottom_left = sort(choices_2, bottom_left_circles)
    sorted_top_right = sort(choices_3, top_right_circles)
    sorted_bottom_right = sort(choices_4, bottom_right_circles)

    # Concatenate the sorted lists from each quadrant
    sorted_circles = sorted_top_left + sorted_bottom_left + sorted_top_right + sorted_bottom_right

    return sorted_top_left, sorted_bottom_left, sorted_top_right, sorted_bottom_right


def sort(column, circles):
    sorted_cols = []

    for k in range(0, len(circles), column):
        col = circles[k:k + column]
        sorted_cols.extend(sorted(col, key=lambda v: v[0]))

    return sorted_cols


def get_shading_percentage(roi):
    total_pixels = roi.size
    shaded_pixels = cv2.countNonZero(roi)
    shading_percentage = (shaded_pixels / total_pixels) * 100
    # print(shading_percentage)

    return shading_percentage


def check(extracted_answers, correct_answers):
    number_of_correct = 0
    number_of_incorrect = 0

    for correct, student in zip(correct_answers, extracted_answers):
        if correct == student:
            number_of_correct += 1
        else:
            number_of_incorrect += 1

    return number_of_correct, number_of_incorrect
