# Pixora Similarity Checker

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-4EA94B?style=for-the-badge&logo=mongodb&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

An intelligent image similarity checker API that prevents duplicate uploads by comparing visual similarity between images. Built with FastAPI and PyTorch, this service uses deep learning to generate and compare image embeddings.

## 🚀 Features

- **Deep Learning-Powered Similarity Detection**: Uses ResNet50 to generate image embeddings
- **Efficient Image Comparison**: Employs cosine similarity to detect visually similar images
- **Duplicate Prevention**: Automatically blocks uploads of images that are too similar to existing ones
- **REST API**: Easy integration with frontend applications or other services
- **MongoDB Integration**: Stores images and their embedding vectors for persistent similarity checking

## 📋 Prerequisites

- Python 3.7+
- MongoDB database

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/RealChAuLa/pixora_similarity_checker.git
   cd pixora_similarity_checker
   ```

2. Set up a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies using the requirements.txt file:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory with the following variables:
   ```
   MONGODB_URI=your_mongodb_connection_string
   DB_NAME=your_database_name
   ```

## 🚀 Running the Application

Start the FastAPI server:
```bash
uvicorn main:app --reload
```

The API will be accessible at `http://127.0.0.1:8000`.

## 📚 API Documentation

Once the server is running, you can access the automatic API documentation:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

### API Endpoints

#### `POST /upload`

Uploads an image and checks for similarity with existing images in the database.

**Request Body:**
```json
{
  "imageBase64": "base64_encoded_image_string",
  "name": "image_name"
}
```

**Responses:**

- `200 OK`: Image successfully uploaded
  ```json
  {
    "message": "NFT saved successfully in MongoDB"
  }
  ```

- `400 Bad Request`: Missing image data
  ```json
  {
    "error": "No imageBase64 data provided"
  }
  ```

- `200 OK` (with error message): Similar image found
  ```json
  {
    "error": "NFT already exists in MongoDB"
  }
  ```

## 🧠 How It Works

1. **Image Processing**: Images are preprocessed and normalized to a standard format
2. **Embedding Generation**: A pre-trained ResNet50 model generates a feature vector (embedding) for each image
3. **Similarity Calculation**: Cosine similarity is used to compare new images with existing ones
4. **Threshold Filtering**: Images with similarity scores above 0.9 are considered duplicates
5. **Storage**: Unique images are stored with their embeddings in MongoDB for future comparison

## 🔧 Technical Details

- **Model**: ResNet50 (pre-trained) with classification layer removed
- **Similarity Metric**: Cosine similarity
- **Duplicate Threshold**: 0.9 (90% similarity)
- **Image Preprocessing**: Resize to 224x224, normalize with ImageNet mean and std
- **Database**: MongoDB for storing images and embeddings
