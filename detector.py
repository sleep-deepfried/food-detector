import os
from flask import Flask, jsonify, request
from flask_cors import CORS
import boto3
import cv2
import numpy as np
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to track camera state
current_camera = None


class RefrigeratorMonitor:
    def __init__(self, region_name="us-east-1"):
        """
        Initialize the RefrigeratorMonitor with AWS Rekognition
        """
        # AWS Rekognition setup
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

        if not aws_access_key_id or not aws_secret_access_key:
            raise ValueError("AWS credentials not found in environment variables")

        self.rekognition_client = boto3.client(
            "rekognition",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
        )

        self.food_categories = {
            "Apple": "Fruits",
            "Banana": "Fruits",
            "Carrot": "Vegetables",
            "Chicken": "Meat",
            "Milk": "Dairy",
        }

    def detect_items_from_frame(self, frame):
        """
        Detect food items in the provided frame using AWS Rekognition
        """
        # Encode the frame to jpg
        _, buffer = cv2.imencode(".jpg", frame)
        if not _:
            logger.error("Failed to encode image")
            return []

        # Convert to binary
        binary_image = buffer.tobytes()

        # Call AWS Rekognition
        response = self.rekognition_client.detect_labels(
            Image={"Bytes": binary_image},
            MaxLabels=20,
            MinConfidence=70,
        )

        logger.info(f"AWS Rekognition response: {response}")

        # Extract relevant detected items
        detected_items = []
        for label in response["Labels"]:
            logger.info(
                f"Detected label: {label['Name']} with confidence: {label['Confidence']}"
            )
            if label["Name"] in self.food_categories:
                detected_items.append(
                    {"name": label["Name"], "confidence": label["Confidence"]}
                )
        return detected_items

    def add_items(self, items):
        """
        Add detected items with expiration and category details
        """
        current_time = datetime.utcnow()
        expiration_days = {
            "Fruits": 7,
            "Vegetables": 10,
            "Meat": 3,
            "Dairy": 5,
        }

        inventory = []
        for item in items:
            category = self.food_categories.get(item["name"], "Other")
            expiration_delta = timedelta(days=expiration_days.get(category, 3))

            new_item = {
                "name": item["name"],
                "category": category,
                "addedAt": current_time.isoformat(),
                "expirationDate": (current_time + expiration_delta).isoformat(),
                "confidence": item["confidence"],
            }
            inventory.append(new_item)
        return inventory


# Create a global monitor instance
monitor = RefrigeratorMonitor()


@app.route("/open-camera", methods=["POST"])
def open_camera():
    """
    Open the camera if it's not already open
    """
    global current_camera
    try:
        # Check if camera is already open
        if current_camera is not None and current_camera.isOpened():
            logger.warning("Camera is already open")
            return jsonify({"status": "warning", "message": "Camera already open"}), 400

        # Open the camera
        current_camera = cv2.VideoCapture(0)

        # Check if camera opened successfully
        if not current_camera.isOpened():
            logger.error("Unable to access the camera")
            current_camera = None
            return jsonify({"error": "Unable to access the camera"}), 500

        logger.info("Camera opened successfully")
        return jsonify({"status": "success", "message": "Camera opened"}), 200

    except Exception as e:
        logger.error(f"Error in /open-camera: {e}")
        # Ensure camera is released in case of an exception
        if current_camera is not None:
            current_camera.release()
            current_camera = None
        return jsonify({"error": str(e)}), 500


@app.route("/add", methods=["GET"])
def add_foods():
    """
    Scan for food items using an already open camera and return the results
    """
    global current_camera
    try:
        # Check if camera is open
        if current_camera is None or not current_camera.isOpened():
            logger.error("Camera is not open. Please open camera first.")
            return (
                jsonify({"error": "Camera is not open. Use /open-camera first."}),
                500,
            )

        # Capture one frame from the camera
        ret, frame = current_camera.read()

        # Ensure the frame is successfully captured before proceeding
        if not ret:
            logger.error("Failed to capture image from camera")
            return jsonify({"error": "Failed to capture image from camera"}), 500

        logger.info("Image captured successfully")

        # Resize the image to a smaller resolution (optional, for debugging)
        frame = cv2.resize(frame, (640, 480))

        # Detect items in the captured frame
        detected_items = monitor.detect_items_from_frame(frame)

        if not detected_items:
            logger.warning("No food items detected")

        # Add detected items to inventory
        added_items = monitor.add_items(detected_items)

        return jsonify({"status": "success", "added_items": added_items}), 200

    except Exception as e:
        logger.error(f"Error in /add: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/close-camera", methods=["POST"])
def close_camera():
    """
    Close the camera if it is currently open
    """
    global current_camera
    try:
        if current_camera is not None and current_camera.isOpened():
            current_camera.release()
            current_camera = None
            logger.info("Camera closed successfully")
            return jsonify({"status": "success", "message": "Camera closed"}), 200
        else:
            logger.warning("No active camera to close")
            return jsonify({"status": "warning", "message": "No active camera"}), 400

    except Exception as e:
        logger.error(f"Error in /close-camera: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
