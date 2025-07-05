#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter, deque
from collections import deque

import cv2
from flask import Flask, request, jsonify

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

from twilio.rest import Client

app = Flask(__name__)
# Global state
locked_hand_index = None
history_length = 16
point_history = deque(maxlen=history_length)
finger_gesture_history = deque(maxlen=history_length)
mode = 0

# Model load
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,  # Default, can be overridden via API params
    max_num_hands=2,
    min_detection_confidence=0.7,  # Default value
    min_tracking_confidence=0.5,  # Default value
)

keypoint_classifier = KeyPointClassifier()
point_history_classifier = PointHistoryClassifier()

# Read labels
with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
    keypoint_classifier_labels = [row[0] for row in csv.reader(f)]
with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
    point_history_classifier_labels = [row[0] for row in csv.reader(f)]

def send_sms(to_number, message_body, account_sid, auth_token, messaging_service_sid=None, from_number=None):
    """
    Sends an SMS using Twilio API.

    Parameters:
        to_number (str): Recipient phone number (e.g., '+919596713921')
        message_body (str): The text message to send
        account_sid (str): Your Twilio Account SID
        auth_token (str): Your Twilio Auth Token
        messaging_service_sid (str, optional): Messaging Service SID (starts with 'MG...')
        from_number (str, optional): Your Twilio phone number (e.g., '+1234567890')

    Returns:
        str: Message SID if sent successfully
    """
    client = Client(account_sid, auth_token)

    try:
        if messaging_service_sid:
            message = client.messages.create(
                messaging_service_sid=messaging_service_sid,
                body=message_body,
                to=to_number
            )
        elif from_number:
            message = client.messages.create(
                from_=from_number,
                body=message_body,
                to=to_number
            )
        else:
            raise ValueError("Either messaging_service_sid or from_number must be provided.")

        return message.sid

    except Exception as e:
        return f"Failed to send SMS: {e}"
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


@app.route('/process_image', methods=['POST'])
def process_image():
    global locked_hand_index, point_history, finger_gesture_history, mode

    # Check if image is provided
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    # Read image from file
    try:
        img_array = np.frombuffer(file.read(), np.uint8)
        image = cv.imdecode(img_array, cv.IMREAD_COLOR)
        if image is None:
            return jsonify({'error': 'Invalid image'}), 400
    except Exception as e:
        return jsonify({'error': f'Error reading image: {str(e)}'}), 400

    # Process image
    image = cv.flip(image, 1)  # Mirror display
    debug_image = copy.deepcopy(image)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True

    response = {'hand_sign': None, 'finger_gesture': None}

    if results.multi_hand_landmarks is not None:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            current_label = handedness.classification[0].label
            if locked_hand_index is None:
                locked_hand_index = current_label
            if current_label != locked_hand_index:
                continue    #ignore other hands

            #Bounding box calculation
            brect = calc_bounding_rect(debug_image, hand_landmarks)
            #Landmark Calculation
            landmark_list = calc_landmark_list(debug_image, hand_landmarks)

            #Conversion to relative coordinates / normalize coordinates
            pre_processed_landmark_list = pre_process_landmark(landmark_list)
            pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)
            #Write to dataset file
            logging_csv(-1, mode, pre_processed_landmark_list, pre_processed_point_history_list)

            #Hand sign classification
            hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
            if hand_sign_id == "not applicable":
                point_history.append(landmark_list[8])
            else:
                point_history.append([0, 0])

            #Fingure gesture classification
            finger_gesture_id = 0
            point_history_len = len(pre_processed_point_history_list)
            if point_history_len == (history_length * 2):
                finger_gesture_id = point_history_classifier(pre_processed_point_history_list)

            #Calculates the gesture IDs in the latest detection
            finger_gesture_history.append(finger_gesture_id)
            most_common_fg_id = Counter(finger_gesture_history).most_common()

            debug_image = draw_info_text(
                debug_image,
                brect,
                handedness,
                keypoint_classifier_labels[hand_sign_id],
                point_history_classifier_labels[most_common_fg_id[0][0]],
            )

            response['hand_sign'] = keypoint_classifier_labels[hand_sign_id]
            response['finger_gesture'] = point_history_classifier_labels[most_common_fg_id[0][0]]
    else:
        point_history.append([0, 0])
        locked_hand_index = None

    debug_image = draw_point_history(debug_image, point_history)

    return jsonify(response)

def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history

def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return

required_sequence = ["Close", "Open", "peace"]
sequence_index = 0

count=0
def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)
    global sequence_index,count
    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        if hand_sign_text == required_sequence[sequence_index]:
            sequence_index += 1  # Move to next step
            print(sequence_index)
        else:
            if sequence_index>0 and hand_sign_text == required_sequence[sequence_index-1] and count<20:
                print("hold")
                count+=1
            else:
                count=0
                sequence_index = 0  # Reset if the sequence is broken

            # If full sequence is detected
        if sequence_index == len(required_sequence):
            cv.putText(image, "Alert: Sequence detected!", (brect[0] + 5, brect[1] - 4),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
            print("Alert: Sequence detected!")  # Trigger alert
            sid = send_sms(
                to_number='+',
                 message_body='Distress signal!',
                 account_sid='',
                 auth_token='',
                 messaging_service_sid='+'
             )
            print("Message SID:", sid)
            sequence_index = 0
        else:

            info_text = info_text + ':' + hand_sign_text
            cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)


    return image

def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
