import os
from flask import Flask, jsonify, request
from flask_cors import CORS
import boto3
import cv2
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float
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
            
            # Common in Filipino cuisine
            "Rice": "Grains",
            "Fish": "Seafood",
            "Tilapia": "Seafood",
            "Bangus (Milkfish)": "Seafood",
            "Galunggong (Mackerel Scad)": "Seafood",
            "Shrimp": "Seafood",
            "Crab": "Seafood",
            "Squid": "Seafood",
            "Coconut": "Fruits",
            "Coconut Milk": "Dairy",
            "Ampalaya (Bitter Gourd)": "Vegetables",
            "Malunggay (Moringa Leaves)": "Vegetables",
            "Kangkong (Water Spinach)": "Vegetables",
            "Sitaw (String Beans)": "Vegetables",
            "Eggplant": "Vegetables",
            "Papaya": "Fruits",
            "Calamansi": "Fruits",
            "Ube (Purple Yam)": "Vegetables",
            "Cassava": "Vegetables",
            "Gabi (Taro)": "Vegetables",
            "Kamote (Sweet Potato)": "Vegetables",
            "Langka (Jackfruit)": "Fruits",
            "Pechay (Bok Choy)": "Vegetables",
            "Chili Pepper": "Vegetables",
            "Tamarind": "Fruits",
            "Ginger": "Vegetables",
            "Vinegar": "Condiments",
            "Soy Sauce": "Condiments",
            "Bagoong (Fermented Shrimp/Fish Paste)": "Condiments",
            "Tuyo (Dried Fish)": "Seafood",
            "Dilis (Anchovies)": "Seafood",
        }

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
    max_labels = int(
        request.args.get("max_labels", 20)
    )  # Default to 20 if not specified
    min_confidence = float(
        request.args.get("min_confidence", 70)
    )  # Default to 70 if not specified

    if current_camera is None or not current_camera.isOpened():
        return jsonify({"error": "Camera is not open. Use /open-camera first."}), 500

    ret, frame = current_camera.read()
    if not ret:
        return jsonify({"error": "Failed to capture image from camera"}), 500

    # Resize frame to match Rekognition's expected input
    frame = cv2.resize(frame, (320, 240))

    # Detect items from the frame using AWS Rekognition
    detected_items = monitor.detect_items_from_frame(frame, max_labels, min_confidence)

    # Process detected items and update the quantities correctly
    updated_items = {}  # Temporary dictionary to hold updated items

    for item in detected_items:
        item_name = item["name"]
        detected_quantity = item["quantity"]
 
        # Debugging: Print the detected item and quantity
        print(f"Detected Item: {item_name}, Quantity: {detected_quantity}")

        # Check if a query parameter for this item exists
        query_quantity = request.args.get(f"quantity_{item_name}")
        if query_quantity:
            print(f"Found query parameter for {item_name}: {query_quantity}")
            detected_quantity = int(
                query_quantity
            )  # Apply the quantity from query parameter

        # Update the item quantity in the current_detected_items dictionary
        if item_name in current_detected_items:
            print(
                f"Updating {item_name} in current_detected_items with quantity {detected_quantity}"
            )
            current_detected_items[item_name]["count"] = detected_quantity
        else:
            print(f"Adding new item {item_name} with quantity {detected_quantity}")
            current_detected_items[item_name] = {
                "name": item_name,
                "count": detected_quantity,
            }

        # Keep track of the updated item in the temporary dictionary
        updated_items[item_name] = current_detected_items[item_name]

    # Return the detected and updated items
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


@app.route("/add", methods=["POST"])
def add_foods():
    global current_detected_items

    if not current_detected_items:
        return jsonify({"error": "No items detected. Use /detect first."}), 400

    # Get query parameters for optional filtering
    filter_category = request.args.get("category")  # Optionally filter by category
    filter_quantity = request.args.get("quantity")  # Quantity filter, if present

    # Prepare items to add from the detected items
    items_to_add = []
    for item_name, item_info in current_detected_items.items():
        # If quantity filter exists, modify the quantity
        if filter_quantity:
            item_info["count"] = int(
                filter_quantity
            )  # Set the quantity to the query parameter value

        # Only include items if they match the category filter (if provided)
        if not filter_category or (
            filter_category and item_info.get("food_type") == filter_category
        ):
            items_to_add.append(
                {
                    "name": item_name,
                    "quantity": item_info["count"],
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

    # Retrieve optional query parameters
    item_name = request.args.get("item_name")  # Name of the item to remove
    quantity_to_remove = int(
        request.args.get("quantity", 1)
    )  # Quantity to remove (default is 1)

    if not current_detected_items:
        return jsonify({"error": "No items detected. Use /detect first."}), 400

    # Use the session context manager for proper session handling
    with SessionLocal() as db:
        query = db.query(FoodInventory)

        # Filter by item name if provided
        if item_name:
            query = query.filter(FoodInventory.food_name == item_name)
        else:
            # Use all detected item names if no specific item_name provided
            item_names = [item["name"] for item in current_detected_items.values()]
            query = query.filter(FoodInventory.food_name.in_(item_names))

        # Retrieve the items to be removed
        items_to_remove = query.all()

        for item in items_to_remove:
            if quantity_to_remove >= item.quantity:
                # If quantity to remove is greater or equal, delete the item
                db.delete(item)
            else:
                # Otherwise, just decrease the quantity
                item.quantity -= quantity_to_remove

        db.commit()
        return (
            jsonify(
                {
                    "status": "success",
                    "removed_items": [
                        {
                            "food_name": item.food_name,
                            "remaining_quantity": item.quantity,
                        }
                        for item in items_to_remove
                    ],
                }
            ),
            200,
        )

@app.route("/close-camera", methods=["POST"])
def close_camera():
    global current_camera
    if current_camera is None or not current_camera.isOpened():
        return jsonify({"status": "warning", "message": "Camera is not open"}), 400
    current_camera.release()
    current_camera = None
    return jsonify({"status": "success", "message": "Camera closed"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
