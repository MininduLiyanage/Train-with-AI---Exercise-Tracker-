# Train-with-AI-Exercise-Tracker

## Project Overview

This project uses computer vision and machine learning techniques to create an exercise tracking system. By leveraging the power of OpenCV (cv2), MediaPipe (mediapipe), we can track and analyze human exercise movements in real time to provide feedback on exercise form and performance.

![Screenshot (679)](https://github.com/MininduLiyanage/Train-with-AI-Exercise-Tracker-python-openCV/assets/73852035/df2adc6f-ce68-4af8-8e48-58ddd8d0ca58)


## Features

1. Real-Time Pose Detection
2. Performance Metrics: Calculate angles, no of reps and accurate completion using progress bar 

## Project Structure
Here’s a brief overview of the main Python files included in this project:

  1. AITrainerProject.py

Purpose: The entry point of the project. It captures video input, performs pose detection, and displays real-time feedback.

Usage: Run this file to start the application and see the pose tracking.

  2. poseModule.py

Purpose: Contains important functions for detecting and tracking skeleton landmarks.

Usage: Imported into AITrainerProject.py.py to provide the core functionalities.

  3. posedetectionbasic.py

Purpose: Contains basics related to pose detection with mediapipe.


## Libraries Used

1. OpenCV
2. mediapipe - Provides pre-trained models for pose estimation and other multimodal data processing.
