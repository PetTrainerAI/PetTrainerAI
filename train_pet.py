import random
import time
import json
import cv2
import tensorflow as tf
import numpy as np
import pyttsx3  # For text-to-speech
import speech_recognition as sr  # For speech-to-text
from flask import Flask, jsonify
from twilio.rest import Client

class PetTrainerAI:
    def __init__(self, pet_name="Buddy", model_path="behavior_model.h5", treat_dispenser_api=None):
        """
        Initialize the PetTrainerAI instance.

        Args:
            pet_name (str): Name of the pet being trained.
            model_path (str): Path to the pre-trained AI model.
            treat_dispenser_api (str, optional): API endpoint for the treat dispenser.
        """
        self.pet_name = pet_name
        self.reward_count = 0
        self.behavior_log = []
        self.target_behavior = None
        self.treat_dispenser_api = treat_dispenser_api

        print(f"PetTrainerAI initialized for {self.pet_name}")

        # Load the AI model for behavior detection
        try:
            self.model = tf.keras.models.load_model(model_path)
            print("AI model loaded successfully.")
        except Exception as e:
            print(f"Error loading AI model: {e}")
            self.model = None

        # Initialize text-to-speech engine
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)

        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()

    def speak(self, text):
        """
        Converts text to speech to communicate with the pet.

        Args:
            text (str): Text to be spoken.
        """
        print(f"[Speaking]: {text}")
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def record_command(self):
        """
        Records the owner's voice command using the microphone.

        Returns:
            str: Transcribed voice command or an error message.
        """
        print("Listening for command...")
        with sr.Microphone() as source:
            try:
                audio_data = self.recognizer.listen(source, timeout=5)
                command = self.recognizer.recognize_google(audio_data)
                print(f"[Recognized Command]: {command}")
                return command
            except sr.UnknownValueError:
                return "Could not understand the command."
            except sr.RequestError as e:
                return f"Speech recognition error: {e}"

    def detect_behavior(self, frame):
        """
        Detects pet behavior using the AI model.

        Args:
            frame (numpy.ndarray): Video frame for AI-based detection.

        Returns:
            str: Detected behavior.
        """
        if self.model is None:
            return "AI model not loaded."

        # Preprocess the frame for the AI model
        frame_resized = cv2.resize(frame, (224, 224))
        frame_normalized = frame_resized / 255.0
        input_data = np.expand_dims(frame_normalized, axis=0)

        # Make predictions
        predictions = self.model.predict(input_data)
        behavior_index = np.argmax(predictions)

        # Map prediction to behavior
        behaviors = {0: "sitting", 1: "barking", 2: "jumping", 3: "lying down", 4: "rolling over"}
        behavior = behaviors.get(behavior_index, "unknown")

        self.behavior_log.append({"behavior": behavior, "timestamp": time.time()})
        return behavior

    def train_pet(self, target_behavior, frame):
        """
        Trains the pet to perform a target behavior.

        Args:
            target_behavior (str): The desired behavior to reinforce.
            frame (numpy.ndarray): Video frame for behavior detection.

        Returns:
            str: Training feedback.
        """
        detected_behavior = self.detect_behavior(frame)

        if detected_behavior == target_behavior:
            self.reward_count += 1
            self.speak(f"Good {self.pet_name}! You performed '{detected_behavior}'. Dispensing treat!")
            if self.treat_dispenser_api:
                self.dispense_treat()
            response = f"Good {self.pet_name}! You performed '{detected_behavior}'. Dispensing treat!"
        else:
            self.speak(f"{self.pet_name}, try again! Detected: '{detected_behavior}'.")
            response = f"{self.pet_name}, try again! Detected: '{detected_behavior}'."
        return response

    def dispense_treat(self):
        """
        Sends a request to the treat dispenser API to dispense a treat.
        """
        if not self.treat_dispenser_api:
            print("No treat dispenser API configured.")
            return

        try:
            import requests
            response = requests.post(self.treat_dispenser_api)
            if response.status_code == 200:
                print("Treat dispensed successfully!")
            else:
                print(f"Failed to dispense treat. Status code: {response.status_code}")
        except Exception as e:
            print(f"Error dispensing treat: {e}")

    def get_training_summary(self):
        """
        Provides a summary of the training session.

        Returns:
            dict: Summary of rewards and behavior log.
        """
        summary = {
            "Pet Name": self.pet_name,
            "Total Rewards": self.reward_count,
            "Behavior Log": self.behavior_log,
        }
        return summary

    def save_training_summary(self, filepath="training_summary.json"):
        """
        Saves the training summary to a JSON file.

        Args:
            filepath (str): Path to save the training summary file.
        """
        summary = self.get_training_summary()
        with open(filepath, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"Training summary saved to {filepath}")

    def reset_training(self):
        """
        Resets the training session.
        """
        self.reward_count = 0
        self.behavior_log = []
        print("Training session has been reset.")

# Flask API for Real-Time Monitoring
app = Flask(__name__)
trainer = None

@app.route('/training-summary', methods=['GET'])
def training_summary():
    if trainer:
        return jsonify(trainer.get_training_summary())
    return jsonify({"error": "Trainer not initialized"})

# Example Usage
if __name__ == "__main__":
    print("Welcome to PetTrainerAI!")
    pet_name = input("Enter your pet's name: ")
    trainer = PetTrainerAI(pet_name=pet_name, treat_dispenser_api="http://localhost:5000/api/dispense")

    # Simulate a video stream
    cap = cv2.VideoCapture(0)  # Use webcam for real-time video

    print(f"Starting training for {pet_name}...\n")

    target_behavior = "sitting"
    print(f"Training target behavior: '{target_behavior}'\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            feedback = trainer.train_pet(target_behavior, frame)
            print(feedback)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("\nTraining session complete!\n")
    summary = trainer.get_training_summary()
    print("Training Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    trainer.save_training_summary()
