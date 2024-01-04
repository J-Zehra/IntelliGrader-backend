import cv2
import numpy as np
import pytesseract
from imutils.object_detection import non_max_suppression


def process(image, parts, correct_answer_indices):
    template_marker = cv2.imread("marker.png", 0)
    template_marker_2 = cv2.imread("marker2.png", 0)
    answer_indices = []
    roll_number = None

    # PREPROCESS IMAGE
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # PROCESS ROLL NUMBER SECTION
    roll_number_section = extract_section(image_gray, template_marker_2)
    if roll_number_section is None:
        return {"status": "error", "message": "Roll Number Not Detected"}

    roll_number_section_blur = cv2.GaussianBlur(roll_number_section, (21, 21), 1)

    # DETECT CIRCLES
    roll_number_circles = cv2.HoughCircles(
        roll_number_section_blur, cv2.HOUGH_GRADIENT, dp=1, minDist=10, param1=80, param2=10, minRadius=5, maxRadius=8
    )

    if roll_number_circles is not None:
        roll_number_circles = np.round(roll_number_circles[0, :]).astype("int")
        detected_roll_number_circles = int(len(roll_number_circles))
        print(f"Detected {detected_roll_number_circles} roll number circles")

        if detected_roll_number_circles != 20:
            return {"status": "error", "message": "Not All Circles Are Detected"}

        sorted_roll_number_circles = sorted(roll_number_circles, key=lambda circle: (circle[1], circle[0]))
        extracted_indices = extract_answer_indices(sorted_roll_number_circles, 10, roll_number_section)
        roll_number = int(''.join(map(str, extracted_indices)))
        print(f"Extracted Roll Number Indices: {extracted_indices}")
        print(f"Roll Number: {roll_number}")

    # PROCESS BUBBLE SECTION
    bubble_section = extract_section(image_gray, template_marker)
    if bubble_section is None:
        return {"status": "error", "message": "Bubble Section Not Detected"}

    bubble_section_blur = cv2.GaussianBlur(bubble_section, (21, 21), 1)

    # DETECT CIRCLES
    circles = cv2.HoughCircles(
        bubble_section_blur, cv2.HOUGH_GRADIENT, dp=1, minDist=10, param1=80, param2=10, minRadius=5, maxRadius=8
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        detected_circles = int(len(circles))
        number_of_circles = int(
            sum(choice["numberOfChoices"] * len(correct_answer_indices) for choice in parts) / len(parts))
        print(f"{number_of_circles} circles")
        print(f"Detected {detected_circles} circles")

        if number_of_circles != detected_circles:
            return {"status": "error", "message": "Not All Circles Are Detected"}

        sorted_top_left, sorted_bottom_left, sorted_top_right, sorted_bottom_right = sort_circles(circles,
                                                                                                  bubble_section,
                                                                                                  parts)

        try:
            choices_1 = parts[0]["numberOfChoices"]
        except IndexError:
            choices_1 = 1

        try:
            choices_2 = parts[1]["numberOfChoices"]
        except IndexError:
            choices_2 = 1

        try:
            choices_3 = parts[2]["numberOfChoices"]
        except IndexError:
            choices_3 = 1

        try:
            choices_4 = parts[3]["numberOfChoices"]
        except IndexError:
            choices_4 = 1

        part_1_answer_indices = extract_answer_indices(sorted_top_left, choices_1, bubble_section)
        part_2_answer_indices = extract_answer_indices(sorted_bottom_left, choices_2, bubble_section)
        part_3_answer_indices = extract_answer_indices(sorted_top_right, choices_3, bubble_section)
        part_4_answer_indices = extract_answer_indices(sorted_bottom_right, choices_4, bubble_section)

        answer_indices = part_1_answer_indices + part_2_answer_indices + part_3_answer_indices + part_4_answer_indices

    number_of_correct, number_of_incorrect, total_score, total_perfect_score = check(answer_indices,
                                                                                     correct_answer_indices, parts)

    return {
        "processed_image": bubble_section,
        "original_image": image,
        "answer_indices": answer_indices,
        "number_of_correct": number_of_correct,
        "number_of_incorrect": number_of_incorrect,
        "total_score": total_score,
        "total_perfect_score": total_perfect_score,
        "roll_number": roll_number,
        "status": "success"
    }


def extract_section(sample_image, template_marker, scale_range=(1, 1.5), scale_step=0.1):
    section = None

    # Generate a range of scales
    scales = np.arange(scale_range[0], scale_range[1] + scale_step, scale_step)

    for scale in scales:
        # Resize the template at the current scale
        resized_template = cv2.resize(template_marker, None, fx=scale, fy=scale)
        (tH, tW) = resized_template.shape[:2]

        # Match the resized template with the sample image
        result = cv2.matchTemplate(sample_image, resized_template, cv2.TM_CCOEFF_NORMED)

        # Set a threshold to consider a match
        threshold = 0.8
        (yCoords, xCoords) = np.where(result >= threshold)

        # initialize our list of rectangles
        rects = []

        # loop over the starting (x, y)-coordinates again
        for (x, y) in zip(xCoords, yCoords):
            # update our list of rectangles
            rects.append((x, y, x + tW, y + tH))

        # apply non-maxima suppression to the rectangles
        pick = non_max_suppression(np.array(rects))
        print("[INFO] {} matched locations *after* NMS".format(len(pick)))

        if len(pick) == 4:
            # loop over the final bounding boxes
            for (startX, startY, endX, endY) in pick:
                # draw the bounding box on the image
                cv2.rectangle(sample_image, (startX, startY), (endX, endY),
                              (255, 0, 0), 3)

            margin = 8

            # Extract the section inside the four detected templates with a margin
            min_x = min([startX for (startX, _, _, _) in pick]) + margin
            min_y = min([startY for (_, startY, _, _) in pick]) + margin
            max_x = max([endX for (_, _, endX, _) in pick]) - margin
            max_y = max([endY for (_, _, _, endY) in pick]) - margin

            # Ensure the coordinates are within bounds
            min_x = max(0, min_x)
            min_y = max(0, min_y)
            max_x = min(sample_image.shape[1], max_x)
            max_y = min(sample_image.shape[0], max_y)

            # Draw the bounding box on the image
            cv2.rectangle(sample_image, (min_x, min_y), (max_x, max_y), (255, 0, 0), 3)

            # Crop the section inside the four detected templates
            section = sample_image[min_y:max_y, min_x:max_x]

            break

    return section


def extract_answer_indices(sorted_circles, number_of_choices, bubble_section):
    answer_indices = []

    for i in range(0, len(sorted_circles), number_of_choices):
        question_circles = sorted_circles[i:i + number_of_choices]
        shaded_index = -1
        shading_count = 0

        for index, (x, y, r) in enumerate(question_circles):
            roi_gray = bubble_section[y - r:y + r, x - r:x + r]
            roi_blur = cv2.GaussianBlur(roi_gray, (21, 21), 1)
            roi_thresh = cv2.adaptiveThreshold(roi_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 41, 40)

            average_intensity = cv2.mean(roi_thresh)[0]
            shading_percentage = (average_intensity / 255) * 100

            if shading_percentage > 40:
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


def sort_circles(circles, cropped_bubble_image, parts):
    try:
        choices_1 = parts[0]["numberOfChoices"]
    except IndexError:
        choices_1 = 1

    try:
        choices_2 = parts[1]["numberOfChoices"]
    except IndexError:
        choices_2 = 1

    try:
        choices_3 = parts[2]["numberOfChoices"]
    except IndexError:
        choices_3 = 1

    try:
        choices_4 = parts[3]["numberOfChoices"]
    except IndexError:
        choices_4 = 1

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


def check(extracted_answers, correct_answers, parts):
    number_of_correct = 0
    number_of_incorrect = 0
    total_score = 0
    total_perfect_score = 0

    current_index = 0

    for part in parts:
        part_size = part['totalNumber']
        part_answers = extracted_answers[current_index:current_index + part_size]
        part_correct = correct_answers[current_index:current_index + part_size]

        for correct, student in zip(part_correct, part_answers):
            if correct == student:
                number_of_correct += 1
                total_score += part['points']
            else:
                number_of_incorrect += 1

        total_perfect_score += part['points'] * part_size
        current_index += part_size

    print(f"Number of Correct: {number_of_correct}")
    print(f"Number of Incorrect: {number_of_incorrect}")
    print(f"Total Score: {total_score}")
    print(f"Total Perfect Score: {total_perfect_score}")

    return number_of_correct, number_of_incorrect, total_score, total_perfect_score
