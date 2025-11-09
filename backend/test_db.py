from pathlib import Path
import os

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

client = MongoClient(os.getenv("MONGO_URI"))
print(client.list_database_names())