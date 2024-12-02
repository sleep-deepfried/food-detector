import os
from flask import Flask, jsonify, request
from flask_cors import CORS
import boto3
import cv2
import numpy as np
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Time, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection setup
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
Base = declarative_base()

# Define the Food Inventory Model
class FoodInventory(Base):
    __tablename__ = "food_inventory"
    inventoryID = Column(Integer, primary_key=True)
    food_name = Column(String(255), nullable=True)
    food_type = Column(String(255), nullable=True)
    entry_date = Column(DateTime(timezone=True), nullable=True)
    best_before = Column(Time(timezone=True), nullable=True)
    confidence = Column(Float, nullable=True)
    quantity = Column(Integer, nullable=True, default=1)

Base.metadata.create_all(bind=engine)

# Global variables to track camera and detected items
current_camera = None
current_detected_items = {}

class RefrigeratorMonitor:
    def __init__(self, region_name="us-east-1"):
        self.rekognition_client = boto3.client(
            "rekognition",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=region_name,
        )
        self.food_categories = {
            "Apple": "Fruits", "Banana": "Fruits", "Carrot": "Vegetables",
            "Chicken": "Meat", "Milk": "Dairy", "Pork": "Meat",
            "Egg": "Poultry", "Beef": "Meat", "Cheese": "Dairy",
            "Potato": "Vegetables", "Tomato": "Vegetables", "Lettuce": "Vegetables",
            "Onion": "Vegetables", "Orange": "Fruits", "Garlic": "Vegetables",
            "Pineapple": "Fruits", "Strawberry": "Fruits", "Bread": "Grains",
            "Mango": "Fruits",
        }

    def detect_items_from_frame(self, frame):
        global current_detected_items
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
            detected_items = {}

            # Iterate over the detected labels
            for label in response["Labels"]:
                if label["Name"] in self.food_categories:
                    item_name = label["Name"]
                    confidence = label["Confidence"]
                    if item_name not in detected_items:
                        detected_items[item_name] = {"name": item_name, "confidence": confidence, "count": 1}
                    else:
                        detected_items[item_name]["count"] += 1

            # Update current_detected_items dictionary with the latest counts
            for item_name, item in detected_items.items():
                if item_name in current_detected_items:
                    current_detected_items[item_name]["count"] += item["count"]
                else:
                    current_detected_items[item_name] = item

            # Return the entire list of detected items with counts (full stack)
            return [{"name": item["name"], "confidence": item["confidence"], "quantity": item["count"]} for item in current_detected_items.values()]

        except Exception as e:
            logger.error(f"Error in AWS Rekognition detection: {e}")
            return []

    def add_items(self, items):
        current_time = datetime.utcnow()
        expiration_days = {"Fruits": 7, "Vegetables": 10, "Meat": 3, "Dairy": 5}
        inventory = []

        # Use the session context manager to ensure proper session management
        with SessionLocal() as db:
            try:
                new_item_added = None  # Keep track if a new item is added
                for item in items:
                    category = self.food_categories.get(item["name"], "Other")
                    expiration_delta = timedelta(days=expiration_days.get(category, 3))
                    quantity = item["quantity"]
                    existing_item = db.query(FoodInventory).filter(FoodInventory.food_name == item["name"]).first()

                    if existing_item:
                        existing_item.quantity += quantity
                    else:
                        new_item = FoodInventory(
                            food_name=item["name"],
                            food_type=category,
                            entry_date=current_time,
                            best_before=(current_time + expiration_delta).time(),
                            confidence=item["confidence"],
                            quantity=quantity
                        )
                        db.add(new_item)  # Add the new item to the session
                        new_item_added = new_item  # Mark new item added

                db.commit()  # Commit all changes at once

                # Only refresh if a new item was added
                if new_item_added:
                    db.refresh(new_item_added)

                # Return all the items (existing or newly added)
                inventory.extend(db.query(FoodInventory).filter(FoodInventory.food_name.in_([item["name"] for item in items])).all())

                return inventory
            except Exception as e:
                db.rollback()  # Rollback the transaction in case of error
                logger.error(f"Error in add_items: {e}")
                raise

# Instantiate the monitor class
monitor = RefrigeratorMonitor()

@app.route("/open-camera", methods=["POST"])
def open_camera():
    global current_camera, current_detected_items
    if current_camera is not None and current_camera.isOpened():
        return jsonify({"status": "warning", "message": "Camera already open"}), 400
    current_camera = cv2.VideoCapture(0)
    current_detected_items = {}
    if not current_camera.isOpened():
        current_camera = None
        return jsonify({"error": "Unable to access the camera"}), 500
    return jsonify({"status": "success", "message": "Camera opened"}), 200

@app.route("/detect", methods=["GET"])
def detect_items():
    global current_camera
    if current_camera is None or not current_camera.isOpened():
        return jsonify({"error": "Camera is not open. Use /open-camera first."}), 500
    ret, frame = current_camera.read()
    if not ret:
        return jsonify({"error": "Failed to capture image from camera"}), 500
    frame = cv2.resize(frame, (640, 480))
    detected_items = monitor.detect_items_from_frame(frame)
    return jsonify({
        "status": "success",
        "detected_items": detected_items  # Now returning the full stack of parsed data
    }), 200

@app.route("/add", methods=["POST"])
def add_foods():
    global current_detected_items
    if not current_detected_items:
        return jsonify({"error": "No items detected. Use /detect first."}), 400
    items = [{"name": item["name"], "confidence": item["confidence"], "quantity": item["count"]} for item in current_detected_items.values()]
    added_items = monitor.add_items(items)
    current_detected_items = {}
    return jsonify({
        "status": "success",
        "added_items": [
            {"inventoryID": item.inventoryID, "food_name": item.food_name, "food_type": item.food_type,
             "entry_date": item.entry_date.isoformat(), "best_before": str(item.best_before),
             "confidence": item.confidence, "quantity": item.quantity} for item in added_items]
    }), 200

@app.route("/remove", methods=["POST"])
def remove_food():
    global current_detected_items
    
    if not current_detected_items:
        return jsonify({"error": "No items detected. Use /detect first."}), 400

    # Use the session context manager to ensure proper session management
    with SessionLocal() as db:
        try:
            # Iterate over detected items and remove or decrease their quantity
            for item in current_detected_items.values():
                item_name = item["name"]
                detected_quantity = item["count"]

                # Check if the item exists in the database
                inventory_item = db.query(FoodInventory).filter(FoodInventory.food_name == item_name).first()
                
                if inventory_item:
                    if inventory_item.quantity > detected_quantity:
                        inventory_item.quantity -= detected_quantity  # Subtract the detected quantity
                    else:
                        db.delete(inventory_item)  # Remove the item from the database if quantity is 0 or less
                else:
                    logger.warning(f"Item '{item_name}' not found in inventory, skipping removal")

            db.commit()  # Commit all changes
            current_detected_items = {}  # Reset detected items after removal
            return jsonify({"status": "success", "message": "Items removed from inventory"}), 200

        except Exception as e:
            db.rollback()  # Rollback in case of error
            logger.error(f"Error in remove_food: {e}")
            return jsonify({"error": "An error occurred while removing the food item"}), 500

@app.route("/close-camera", methods=["POST"])
def close_camera():
    global current_camera, current_detected_items
    if current_camera is None:
        return jsonify({"status": "warning", "message": "Camera is not open"}), 400
    current_camera.release()
    current_camera = None
    current_detected_items = {}
    return jsonify({"status": "success", "message": "Camera closed"}), 200

if __name__ == "__main__":
    app.run(debug=True)
