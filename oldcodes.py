# changes in tags (old)
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