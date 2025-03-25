from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from safetensors.torch import load_file
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import cv2
import numpy as np
from PIL import Image
import requests
import re
from googleapiclient.discovery import build
from ultralytics import YOLO
from io import BytesIO
from pytube import YouTube

# ============== SETUP ==============
app = FastAPI()

# YouTube API Setup (Replace with your API key)
API_KEY = "AIzaSyA8mlZCYg4j1wPqAt-xyLZ45ahDU_T4sIg"
youtube = build("youtube", "v3", developerKey=API_KEY)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("C:/roberta_finetuned")

# Load fine-tuned model
roberta_model = AutoModelForSequenceClassification.from_pretrained("C:/roberta_finetuned")

# Load weights
ROBERTA_MODEL_PATH = r"C:\roberta_finetuned\model.safetensors"
state_dict = load_file(ROBERTA_MODEL_PATH)
roberta_model.load_state_dict(state_dict)

roberta_model.eval()
print("Model loaded successfully!")


# ============== INPUT SCHEMA ==============
class VideoRequest(BaseModel):
    video_url: str


# ============== YOUTUBE DATA FETCH ==============
def get_video_data(video_id):
    """Fetches title, description, and comments for a given video ID."""
    try:
        # Get video details
        video_request = youtube.videos().list(part="snippet", id=video_id)
        video_response = video_request.execute()
        if not video_response["items"]:
            return None

        video = video_response["items"][0]["snippet"]
        title = video["title"]
        description = video.get("description", "")
        thumbnail_url = video["thumbnails"]["high"]["url"]

        # Get comments
        comments = get_video_comments(video_id)

        return {"title": title, "description": description, "comments": comments, "thumbnail_url": thumbnail_url}

    except Exception as e:
        print(f"Error fetching video data: {e}")
        return None


def get_video_comments(video_id, max_comments=10):
    """Fetches top YouTube comments for a given video ID."""
    comments = []
    try:
        request = youtube.commentThreads().list(
            part="snippet", videoId=video_id, maxResults=max_comments, textFormat="plainText"
        )
        response = request.execute()

        for item in response.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)

    except Exception:
        comments.append("No comments found")  # If comments are disabled

    return " ".join(comments)


# ============== TEXT CLASSIFICATION WITH ROBERTA ==============
def classify_text_with_roberta(text):
    """Uses RoBERTa to classify text as Safe, Neutral, or Harmful."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = roberta_model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    # Mapping output labels
    label_map = {0: "Safe", 1: "Neutral", 2: "Harmful"}
    return label_map[predicted_class]


# Load YOLO model
YOLO_MODEL_PATH = r"C:\data\runs\detect\train2\weights\best.pt"  # Path to your trained YOLO model
yolo_model = YOLO(YOLO_MODEL_PATH)

def get_youtube_thumbnail(video_url):
    """Fetch the best available YouTube thumbnail."""
    video_id = extract_video_id(video_url)
    if not video_id:
        raise Exception("Invalid YouTube URL")

    # Try different resolutions
    resolutions = ["maxresdefault", "hqdefault", "mqdefault", "default"]

    for res in resolutions:
        thumbnail_url = f"https://img.youtube.com/vi/{video_id}/{res}.jpg"
        response = requests.get(thumbnail_url)

        if response.status_code == 200:
            return thumbnail_url  # ✅ Return the URL instead of saving the image

    raise Exception("Failed to download any thumbnail")

def classify_thumbnail(video_url):
    """Downloads the YouTube thumbnail and classifies it as Safe or Harmful using YOLO."""
    try:
        # Get thumbnail URL
        thumbnail_url = get_youtube_thumbnail(video_url)

        # Download the image
        response = requests.get(thumbnail_url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")  # Convert to RGB

        # Run YOLO detection
        results = yolo_model(img)

        # Extract detected objects (only class names)
        detected_objects = [yolo_model.names[int(box.cls)] for box in results[0].boxes]

        # ✅ Only check if "unsafe" is detected
        is_unsafe = "unsafe" in detected_objects

        return "Harmful" if is_unsafe else "Safe"

    except Exception as e:
        print(f"Error processing thumbnail: {e}")
        return "Safe"  # Default to Safe if there's an issue


# ============== FINAL VIDEO CLASSIFICATION ==============
def classify_video(title, description, comments, thumbnail_url):
    """Combines text & image classifications to determine final video safety."""
    title_class = classify_text_with_roberta(title)
    description_class = classify_text_with_roberta(description)
    comments_class = classify_text_with_roberta(comments)
    thumbnail_class = classify_thumbnail(thumbnail_url)

    classifications = [title_class, description_class, comments_class, thumbnail_class]

    # If at least 2 elements are Harmful, classify the video as Harmful
    if classifications.count("Harmful") >= 1:
        return "Harmful"
    elif classifications.count("Neutral") >= 2:
        return "Neutral"
    else:
        return "Safe"


# ============== API ENDPOINT ==============
def extract_video_id(url):
    """Extracts video ID from a YouTube URL."""
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    return match.group(1) if match else None


@app.post("/classify/")
async def classify_video_api(request: VideoRequest):
    """API endpoint to classify a YouTube video as Safe, Neutral, or Harmful."""
    video_id = extract_video_id(request.video_url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    # Fetch video metadata
    video_data = get_video_data(video_id)
    if not video_data:
        raise HTTPException(status_code=404, detail="Video metadata not found")

    # Classify video
    classification = classify_video(
        video_data["title"], video_data["description"], video_data["comments"], video_data["thumbnail_url"]
    )

    return {
        "video_id": video_id,
        "classification": classification
    }
