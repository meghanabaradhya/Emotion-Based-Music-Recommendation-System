# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 17:46:17 2024
@author: Dhrumit Patel
"""

from keras.models import load_model
from keras.metrics import MeanSquaredError
from time import sleep
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from keras_preprocessing.image import img_to_array
from keras_preprocessing import image
import cv2

import numpy as np

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
emotion_model = load_model('emotion_detection_model_50epochs.h5', custom_objects={'mse': MeanSquaredError()})
# age_model = load_model('age_model_3epochs.h5', custom_objects={'mse': MeanSquaredError()})
# gender_model = load_model('gender_model_3epochs.h5', custom_objects={'mse': MeanSquaredError()})

class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
gender_labels = ['Male', 'Female']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    labels = []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        # Get image ready for prediction
        roi = roi_gray.astype('float') / 255.0  # Scaling the image
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)  # Expand dims to get it ready for prediction (1, 48, 48, 1)

        preds = emotion_model.predict(roi)[0]  # One hot encoded result for 7 classes
        label = class_labels[preds.argmax()]  # Find the label

        label_position = (x, y)
        cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        emotion_to_genre = {
            'happy': 'pop', 'sad': 'acoustic', 'angry': 'rock', 'relaxed': 'chill', 'neutral': 'indie'
        }
        while(label):
            import spotipy
            from spotipy.oauth2 import SpotifyClientCredentials
            import os

            # Securely retrieve client ID and secret from environment variables
            client_id = "320f7bdf50ac4f7b8381e42e88c1a022"
            client_secret = "fc5db3910c8b44659f4184fa027f3ce8"

            if not client_id or not client_secret:
                raise ValueError("Please set the SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET environment variables.")

            # Initialize the Spotify API client
            sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
                client_id=client_id, client_secret=client_secret
            ))

            results = sp.search(q=label, limit=5)
            tracks = results['tracks']['items']
            print(tracks)
            for song in tracks:
                print(f"- {song['name']} by {song['artists'][0]['name']}")


    cv2.imshow('Live Face Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()