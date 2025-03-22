import os
from flask import Flask, jsonify, request
from flask_cors import CORS
import boto3
import cv2
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float
from sqlalchemy.orm import sessionmaker, scoped_session, declarative_base
import csv

# Load environment variables
load_dotenv()
# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection setup
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = scoped_session(
    sessionmaker(autocommit=False, autoflush=False, bind=engine)
)
Base = declarative_base()


# Define the Food Inventory Model
class FoodInventory(Base):
    __tablename__ = "food_inventory"
    inventoryid = Column(Integer, primary_key=True)
    food_name = Column(String(255), nullable=True)
    food_type = Column(String(255), nullable=True)
    entry_date = Column(DateTime(timezone=True), nullable=True)
    best_before = Column(DateTime(timezone=True), nullable=True)
    confidence = Column(Float, nullable=True)
    quantity = Column(Integer, nullable=True, default=1)


Base.metadata.create_all(bind=engine)

# Global variables to track camera and detected items
current_camera = None
current_detected_items = {}


class RefrigeratorMonitor:
    def __init__(self, region_name="ap-southeast-1"):
        self.rekognition_client = boto3.client(
            "rekognition",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=region_name,
        )

        self.food_categories = self.load_food_categories("food_categories.csv")

    def load_food_categories(self, csv_file):
        categories = {}
        try:
            with open(csv_file, mode="r") as file:
                reader = csv.reader(file)
                for row in reader:
                    categories[row[0]] = row[1]
        except Exception as e:
            logger.error(f"Error in loading food categories: {e}")
        return categories

    def detect_items_from_frame(self, frame, max_labels=10, min_confidence=80):
        global current_detected_items
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            logger.error("Failed to encode image")
            return []
        binary_image = buffer.tobytes()
        try:
            response = self.rekognition_client.detect_labels(
                Image={"Bytes": binary_image},
                MaxLabels=max_labels,
                MinConfidence=min_confidence,
            )
            detected_items = {}

            # Iterate over the detected labels
            for label in response["Labels"]:
                if label["Name"] in self.food_categories:
                    item_name = label["Name"]
                    confidence = label["Confidence"]
                    if item_name not in detected_items:
                        detected_items[item_name] = {
                            "name": item_name,
                            "confidence": confidence,
                            "count": 1,
                        }
                    else:
                        detected_items[item_name]["count"] += 1

            # Update current_detected_items dictionary with the latest counts
            for item_name, item in detected_items.items():
                if item_name in current_detected_items:
                    current_detected_items[item_name]["count"] += item["count"]
                else:
                    current_detected_items[item_name] = item

            # Return the entire list of detected items with counts (full stack)
            return [
                {
                    "name": item["name"],
                    "confidence": item["confidence"],
                    "quantity": item["count"],
                }
                for item in current_detected_items.values()
            ]

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
                    existing_item = (
                        db.query(FoodInventory)
                        .filter(FoodInventory.food_name == item["name"])
                        .first()
                    )

                    if existing_item:
                        existing_item.quantity += quantity
                    else:
                        new_item = FoodInventory(
                            food_name=item["name"],
                            food_type=category,
                            entry_date=current_time,
                            best_before=(current_time + expiration_delta),
                            confidence=item["confidence"],
                            quantity=quantity,
                        )
                        db.add(new_item)  # Add the new item to the session
                        new_item_added = new_item  # Mark new item added

                db.commit()  # Commit all changes at once

                # Only refresh if a new item was added
                if new_item_added:
                    db.refresh(new_item_added)

                # Return all the items (existing or newly added)
                inventory.extend(
                    db.query(FoodInventory)
                    .filter(
                        FoodInventory.food_name.in_([item["name"] for item in items])
                    )
                    .all()
                )

                return inventory
            except Exception as e:
                db.rollback()  # Rollback the transaction in case of error
                logger.error(f"Error in add_items: {e}")
                raise


# Instantiate the monitor class
monitor = RefrigeratorMonitor()


@app.route("/detect", methods=["GET"])
def detect_items():
    global current_camera, current_detected_items
    max_labels = int(request.args.get("max_labels", 20))
    min_confidence = float(request.args.get("min_confidence", 70))

    try:
        # Open the camera if it's not already open
        if current_camera is None or not current_camera.isOpened():
            logger.info("Opening camera for detection")
            # Try opening with different camera indices
            for camera_index in [0, 1, 2]:
                try:
                    if current_camera is not None:
                        current_camera.release()  # Release any previous camera object
                        import time

                        time.sleep(1)  # Small delay to ensure proper release

                    current_camera = cv2.VideoCapture(camera_index)
                    if current_camera.isOpened():
                        logger.info(
                            f"Successfully opened camera at index {camera_index}"
                        )
                        break
                except Exception as e:
                    logger.error(f"Failed to open camera at index {camera_index}: {e}")

            # If we couldn't open any camera
            if current_camera is None or not current_camera.isOpened():
                return jsonify({"error": "Unable to access any camera"}), 500

        # Capture a frame
        ret, frame = current_camera.read()
        if not ret:
            return jsonify({"error": "Failed to capture image from camera"}), 500

        # Resize frame for Rekognition
        frame = cv2.resize(frame, (320, 240))

        # Detect items
        detected_items = monitor.detect_items_from_frame(
            frame, max_labels, min_confidence
        )

        # Process detected items and update quantities
        updated_items = {}
        for item in detected_items:
            item_name = item["name"]
            detected_quantity = item["quantity"]

            # Check for query parameter override
            query_quantity = request.args.get(f"quantity_{item_name}")
            if query_quantity:
                detected_quantity = int(query_quantity)

            # Update the item quantity
            if item_name in current_detected_items:
                current_detected_items[item_name]["count"] = detected_quantity
            else:
                current_detected_items[item_name] = {
                    "name": item_name,
                    "count": detected_quantity,
                    "confidence": item["confidence"],
                }

            updated_items[item_name] = current_detected_items[item_name]

        # Close the camera after detection
        if current_camera is not None and current_camera.isOpened():
            current_camera.release()
            current_camera = None
            logger.info("Camera closed after detection")

        # Return results
        return (
            jsonify(
                {
                    "status": "success",
                    "detected_items": [
                        {
                            "name": item["name"],
                            "confidence": item["confidence"],
                            "quantity": item["count"],
                        }
                        for item in updated_items.values()
                    ],
                }
            ),
            200,
        )

    except Exception as e:
        # Ensure camera is closed in case of errors
        if current_camera is not None:
            current_camera.release()
            current_camera = None

        logger.error(f"Error in detect_items: {e}")
        return jsonify({"error": f"Failed to detect items: {str(e)}"}), 500


@app.route("/add", methods=["POST"])
def add_foods():
    global current_detected_items

    if not current_detected_items:
        return jsonify({"error": "No items detected. Use /detect first."}), 400

    # Get query parameters for optional filtering
    filter_category = request.args.get("category")  # Optionally filter by category

    # Prepare items to add from the detected items, with manually updated quantities from frontend
    items_to_add = []
    for item_name, item_info in current_detected_items.items():
        # Check if there's a manually specified quantity for this item
        manual_quantity = request.args.get(f"quantity_{item_name}")

        # Use manually specified quantity if available, otherwise use detected count
        quantity = (
            int(manual_quantity) if manual_quantity is not None else item_info["count"]
        )

        # Only include items if they match the category filter (if provided)
        if not filter_category or (
            filter_category and item_info.get("food_type") == filter_category
        ):
            # Only add items with quantity > 0
            if quantity > 0:
                items_to_add.append(
                    {
                        "name": item_name,
                        "quantity": quantity,
                        "confidence": item_info["confidence"],
                    },
                )

    if not items_to_add:
        return jsonify({"error": "No items to add based on the filter provided."}), 400

    # Add items to the database using the monitor's add_items method
    added_items = monitor.add_items(items_to_add)

    # Clear the current detected items after adding them
    current_detected_items.clear()

    return (
        jsonify(
            {
                "status": "success",
                "added_items": [
                    {
                        "inventoryid": item.inventoryid,
                        "food_name": item.food_name,
                        "food_type": item.food_type,
                        "entry_date": item.entry_date.isoformat(),
                        "best_before": str(item.best_before),
                        "confidence": item.confidence,
                        "quantity": item.quantity,
                    }
                    for item in added_items
                ],
            }
        ),
        200,
    )


@app.route("/remove", methods=["POST"])
def remove_food():
    global current_detected_items

    if not current_detected_items:
        return jsonify({"error": "No items detected. Use /detect first."}), 400

    # Get query parameters for optional filtering
    filter_category = request.args.get("category")  # Optionally filter by category

    # Prepare items to remove with manually updated quantities from frontend
    items_to_remove = []
    for item_name, item_info in current_detected_items.items():
        # Check if there's a manually specified quantity for this item
        manual_quantity = request.args.get(f"quantity_{item_name}")

        # Use manually specified quantity if available, otherwise use detected count
        quantity = (
            int(manual_quantity) if manual_quantity is not None else item_info["count"]
        )

        # Only include items if they match the category filter (if provided)
        if not filter_category or (
            filter_category and item_info.get("food_type") == filter_category
        ):
            # Only remove items with quantity > 0
            if quantity > 0:
                items_to_remove.append({"name": item_name, "quantity": quantity})

    if not items_to_remove:
        return (
            jsonify({"error": "No items to remove based on the filter provided."}),
            400,
        )

    # Use the session context manager for proper session handling
    with SessionLocal() as db:
        try:
            removed_items = []
            for item in items_to_remove:
                item_name = item["name"]
                quantity_to_remove = item["quantity"]

                # Find the item in the database
                db_item = (
                    db.query(FoodInventory)
                    .filter(FoodInventory.food_name == item_name)
                    .first()
                )

                if db_item:
                    # Track the item and its status
                    item_status = {
                        "food_name": db_item.food_name,
                        "initial_quantity": db_item.quantity,
                    }

                    if quantity_to_remove >= db_item.quantity:
                        # If quantity to remove is greater or equal, delete the item
                        db.delete(db_item)
                        item_status["status"] = "deleted"
                        item_status["remaining_quantity"] = 0
                    else:
                        # Otherwise, just decrease the quantity
                        db_item.quantity -= quantity_to_remove
                        item_status["status"] = "updated"
                        item_status["remaining_quantity"] = db_item.quantity

                    removed_items.append(item_status)

            # Commit all changes
            db.commit()

            # Clear the current detected items after removing them
            current_detected_items.clear()

            return jsonify({"status": "success", "removed_items": removed_items}), 200

        except Exception as e:
            db.rollback()  # Rollback in case of error
            logger.error(f"Error in remove_food: {e}")
            return jsonify({"error": f"Failed to remove items: {str(e)}"}), 500


@app.route("/clear", methods=["POST"])
def clear_items():
    global current_detected_items
    current_detected_items.clear()
    return jsonify({"status": "success", "message": "Detected items cleared"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
