# AI-Model-for-YouTube-Content-Moderation
An AI-based content moderation system that can analyze and classify YouTube video metadata (title, description, comments) to determine if a video is safe or harmful based on predefined categories. 

Task 1: Dataset Preparation & Preprocessing
- Use YouTube Data API to collect metadata from at least 50 YouTube videos based on different categories (education, gaming, violence, explicit, news).
- Preprocess text data (title, description, comments) by removing stopwords, lemmatization, and tokenization.
- Perform Sentiment Analysis to classify content as safe or harmful.

Task 2: AI Model Development (NLP-based Classification)
- Train a Text Classification Model using RoBERTa/BERT or LSTM to classify video metadata as Safe, Harmful, or Neutral.
- Fine-tune the model on an open-source dataset (such as the Jigsaw Toxic Comment Dataset).
- Evaluate the model using accuracy, precision, recall, and F1-score.

Task 3: Computer Vision & Image Analysis
- Extract YouTube video thumbnails and classify them using YOLOv5 or OpenCV for harmful imagery detection.
- Implement a binary classification (Safe/Unsafe) using a Convolutional Neural Network (CNN).

Task 4: Model Deployment (Basic API Creation)
- Deploy the AI model as a FastAPI Flask application.
- Create an API endpoint where users can submit a YouTube video link, and the model will return a classification (Safe, Harmful, Neutral).


