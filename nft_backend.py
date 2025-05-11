from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv
import uuid
import base64
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
from io import BytesIO

load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pre-trained ResNet-50 model
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove classification layer
model.eval()

# Preprocessing pipeline for images
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# MongoDB configuration
mongodb_uri = os.getenv("MONGODB_URI", 'mongodb+srv://chalanadevinda:Chalana2003@pixora.hnft1r3.mongodb.net/?retryWrites=true&w=majority&appName=Pixora')
db_name = os.getenv("DB_NAME", "pixora_db")
client = AsyncIOMotorClient(mongodb_uri)
db = client[db_name]

async def get_image_embedding(image_base64: str):
    """Converts a Base64 image to a feature embedding using ResNet-50."""
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_bytes)).convert("RGB")  # Directly process in memory
    image_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model(image_tensor).flatten().numpy()
    return embedding

def cosine_similarity(vec1, vec2):
    """Computes cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

@app.post("/upload")
async def upload_image(request: Request):
    """Handles image uploads and checks for similarity with existing NFTs."""
    data = await request.json()
    image_base64 = data.get("imageBase64")
    image_name = data.get("name", f"{uuid.uuid4()}.png")

    if not image_base64:
        raise HTTPException(status_code=400, detail="No imageBase64 data provided")

    # Generate embedding for the uploaded image
    image_embedding = await get_image_embedding(image_base64)

    try:
        # Check similarity with existing NFTs
        existing_nfts = await db["NFT"].find().to_list(None)

        for nft in existing_nfts:
            existing_embedding = nft.get("embedding")

            # Skip if the existing embedding is missing
            if existing_embedding is None:
                continue

            # Compute similarity
            similarity = cosine_similarity(image_embedding, existing_embedding)
            if similarity > 0.9:  # Threshold for similarity
                return {"error": "NFT already exists in MongoDB"}

        # Save image and embedding to MongoDB as an NFT
        await db["NFT"].insert_one({
            "name": image_name,
            "imageBase64": image_base64,
            "embedding": image_embedding.tolist()
        })

        return {"message": "NFT saved successfully in MongoDB"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")