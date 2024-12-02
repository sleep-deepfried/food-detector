import os
from flask import Flask, jsonify, request
from flask_cors import CORS
import boto3
import cv2
import numpy as np
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv

# SQLAlchemy imports
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Time, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.exc import SQLAlchemyError

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection setup
try:
    DATABASE_URL = os.getenv('DATABASE_URL')  # e.g., 'postgresql://username:password@localhost/dbname'
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable is not set")

    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))

    # Create a base class for declarative models
    Base = declarative_base()

    # Define the Food Inventory Model with updated columns
    class FoodInventory(Base):
        __tablename__ = "food_inventory"

        inventoryID = Column(Integer, primary_key=True)
        food_name = Column(String(255), nullable=True)
        food_type = Column(String(255), nullable=True)
        entry_date = Column(DateTime(timezone=True), nullable=True)
        best_before = Column(Time(timezone=True), nullable=True)
        confidence = Column(Float, nullable=True)
        quantity = Column(Integer, nullable=True, default=1)  # New quantity column

    # Create tables in the database
    Base.metadata.create_all(bind=engine)

except Exception as e:
    logger.error(f"Database connection error: {e}")
    raise

# Global variable to track camera state
current_camera = None


class RefrigeratorMonitor:
    def __init__(self, region_name="us-east-1"):
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
            "Pork": "Meat",
            "Egg": "Poultry",
            "Beef": "Meat",
            "Cheese": "Dairy",
            "Potato": "Vegetables",
            "Tomato": "Vegetables",
            "Lettuce": "Vegetables",
            "Onion": "Vegetables",
            "Orange": "Fruits",
            "Garlic": "Vegetables",
            "Pineapple": "Fruits",
            "Strawberry": "Fruits",
            "Bread": "Grains",
            "Mango": "Fruits",
        }

    def detect_items_from_frame(self, frame):
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            logger.error("Failed to encode image")
            return []

        binary_image = buffer.tobytes()

        try:
            response = self.rekognition_client.detect_labels(
                Image={"Bytes": binary_image},
                MaxLabels=20,
                MinConfidence=70,
            )

            logger.info(f"AWS Rekognition response: {response}")
            detected_items = []
            for label in response["Labels"]:
                logger.info(f"Detected label: {label['Name']} with confidence: {label['Confidence']}")
                if label["Name"] in self.food_categories:
                    detected_items.append({"name": label["Name"], "confidence": label["Confidence"]})
            return detected_items
        except Exception as e:
            logger.error(f"Error in AWS Rekognition detection: {e}")
            return []

    def add_items(self, items, quantities=None):
        current_time = datetime.utcnow()
        expiration_days = {
            "Fruits": 7,
            "Vegetables": 10,
            "Meat": 3,
            "Dairy": 5,
        }

        inventory = []
        db = SessionLocal()
        for i, item in enumerate(items):
            category = self.food_categories.get(item["name"], "Other")
            expiration_delta = timedelta(days=expiration_days.get(category, 3))

            # Use provided quantity or default to 1
            quantity = quantities[i] if quantities and i < len(quantities) else 1

            # Check if the item already exists in the database
            existing_item = db.query(FoodInventory).filter(FoodInventory.food_name == item["name"]).first()

            if existing_item:
                # If it exists, update the quantity
                existing_item.quantity += quantity
                db.commit()
                inventory.append(existing_item)
                logger.info(f"Updated existing item: {item['name']} quantity: {existing_item.quantity}")
            else:
                # If it doesn't exist, add a new item
                new_item = FoodInventory(
                    food_name=item["name"],
                    food_type=category,
                    entry_date=current_time,
                    best_before=(current_time + expiration_delta).time(),  # Extract time component
                    confidence=item["confidence"],
                    quantity=quantity
                )
                db.add(new_item)
                db.commit()
                db.refresh(new_item)
                inventory.append(new_item)
                logger.info(f"Added new item: {item['name']} quantity: {quantity}")
        db.close()
        return inventory


monitor = RefrigeratorMonitor()


@app.route("/open-camera", methods=["POST"])
def open_camera():
    global current_camera
    try:
        if current_camera is not None and current_camera.isOpened():
            logger.warning("Camera is already open")
            return jsonify({"status": "warning", "message": "Camera already open"}), 400

        current_camera = cv2.VideoCapture(0)
        if not current_camera.isOpened():
            logger.error("Unable to access the camera")
            current_camera = None
            return jsonify({"error": "Unable to access the camera"}), 500

        logger.info("Camera opened successfully")
        return jsonify({"status": "success", "message": "Camera opened"}), 200

    except Exception as e:
        logger.error(f"Error in /open-camera: {e}")
        if current_camera is not None:
            current_camera.release()
            current_camera = None
        return jsonify({"error": str(e)}), 500


@app.route("/add", methods=["GET", "POST"])
def add_foods():
    global current_camera
    db = SessionLocal()
    try:
        # Handle both GET and POST requests
        if request.method == 'POST':
            # If manual addition via POST
            items = request.json.get('items', [])
            quantities = request.json.get('quantities', [])
            if not items:
                return jsonify({"error": "No items provided"}), 400
        else:
            # Camera-based detection for GET request
            if current_camera is None or not current_camera.isOpened():
                logger.error("Camera is not open. Please open camera first.")
                return jsonify({"error": "Camera is not open. Use /open-camera first."}), 500

            ret, frame = current_camera.read()
            if not ret:
                logger.error("Failed to capture image from camera")
                return jsonify({"error": "Failed to capture image from camera"}), 500

            logger.info("Image captured successfully")
            frame = cv2.resize(frame, (640, 480))
            detected_items = monitor.detect_items_from_frame(frame)

            if not detected_items:
                logger.warning("No food items detected")
                return jsonify({"status": "warning", "message": "No items detected"}), 200

            items = detected_items
            # Default to 1 for each detected item
            quantities = [1] * len(items)

        # Add items to database
        added_items = monitor.add_items(items, quantities)
        db_items = []
        for item in added_items:
            db_items.append(item)

        logger.info(f"Successfully added/updated {len(db_items)} items in the database")

        return jsonify({
            "status": "success",
            "added_items": [
                {
                    "inventoryID": item.inventoryID,
                    "food_name": item.food_name,
                    "food_type": item.food_type,
                    "entry_date": item.entry_date.isoformat() if item.entry_date else None,
                    "best_before": str(item.best_before) if item.best_before else None,
                    "confidence": item.confidence,
                    "quantity": item.quantity
                } for item in db_items
            ]
        }), 200

    except SQLAlchemyError as db_error:
        db.rollback()
        logger.error(f"Database error in /add: {db_error}")
        return jsonify({"error": "Database error occurred", "details": str(db_error)}), 500

    except Exception as e:
        logger.error(f"Error in /add: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/close-camera", methods=["POST"])
def close_camera():
    global current_camera
    try:
        if current_camera is None:
            return jsonify({"status": "warning", "message": "Camera is not open"}), 400

        current_camera.release()
        current_camera = None
        return jsonify({"status": "success", "message": "Camera closed"}), 200

    except Exception as e:
        logger.error(f"Error in /close-camera: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
