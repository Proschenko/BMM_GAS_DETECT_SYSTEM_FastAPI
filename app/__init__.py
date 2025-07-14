# __init__.py
import os
from dotenv import load_dotenv

load_dotenv()
os.makedirs("uploads", exist_ok=True)