
# Emotion-Based Music Recommendation System

This project implements an emotion-based music recommendation system that uses facial recognition and emotion classification to suggest suitable music based on the user's current emotional state. The system leverages a pre-trained emotion detection model for classifying emotions and the Spotify API for music recommendations.

## Features

- **Emotion Detection**: Classifies the user's emotion using a pre-trained deep learning model.
- **Music Recommendation**: Recommends music based on the detected emotion by querying the Spotify API.
- **Real-time Interaction**: Captures facial expressions from a webcam in real-time.

## Requirements

Ensure you have the following Python libraries installed:

- `keras`
- `spotipy`
- `opencv-python`
- `numpy`
- `keras_preprocessing`
- `tensorflow`

You can install the dependencies using:

```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── live_detection.py               # Main script to capture webcam feed, detect emotions, and recommend music
├── emotion_detection_model_50epochs.h5  # Pre-trained emotion detection model
├── haarcascade_frontalface_default.xml  # Haar cascade file for face detection
├── requirements.txt                # List of Python dependencies
├── README.md                       # Project documentation
```

## How It Works

1. **Face Detection**: The system uses OpenCV's Haar Cascade Classifier to detect faces from the webcam feed.
2. **Emotion Detection**: A pre-trained emotion detection model predicts emotions like `Happy`, `Sad`, `Angry`, etc., based on facial expressions.
3. **Music Recommendation**: The detected emotion is mapped to a specific music genre, and music recommendations are fetched from Spotify using the `spotipy` API.

## Configuration

Before running the script, you need to set up your Spotify API credentials securely:

1. Create an app on the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/applications).
2. Get the **Client ID** and **Client Secret** for the app.
3. Set them as environment variables in your terminal or use a `.env` file (recommended for security purposes).

```bash
export SPOTIPY_CLIENT_ID="your-client-id"
export SPOTIPY_CLIENT_SECRET="your-client-secret"
```

Alternatively, replace the `client_id` and `client_secret` directly in the code (not recommended for production use).

## How to Run

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/emotion-music-recommendation.git
    cd emotion-music-recommendation
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the detection and recommendation system:

    ```bash
    python live_detection.py
    ```

4. The system will activate the webcam, detect your face, classify your emotion, and suggest music based on that emotion. Press `q` to exit.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
