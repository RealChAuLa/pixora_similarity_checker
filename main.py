from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
# Removing motor import
# from motor.motor_asyncio import AsyncIOMotorClient
# Adding direct pymongo import instead
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import base64
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
from io import BytesIO
import uvicorn

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove classification layer
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

mongodb_uri = os.getenv("MONGODB_URI")
db_name = os.getenv("DB_NAME")

if not mongodb_uri or not db_name:
    raise Exception("MongoDB URI and Database Name must be set in the .env file.")

# Changed from AsyncIOMotorClient to regular MongoClient
client = MongoClient(mongodb_uri)
db = client[db_name]

async def get_image_embedding(image_base64: str):
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model(image_tensor).flatten().numpy()
    return embedding

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

@app.post("/upload")
async def upload_image(request: Request):
    data = await request.json()
    image_base64 = data.get("imageBase64")
    image_name = data.get("name")

    if not image_base64:
        raise HTTPException(status_code=400, detail="No imageBase64 data provided")

    image_embedding = await get_image_embedding(image_base64)

    try:
        # Changed from motor's async find() to PyMongo's synchronous find()
        existing_nfts = list(db["NFT"].find())

        for nft in existing_nfts:
            existing_embedding = nft.get("embedding")

            if existing_embedding is None:
                continue

            similarity = cosine_similarity(image_embedding, existing_embedding)
            if similarity > 0.9:
                return {"error": "NFT already exists in MongoDB"}

        # Changed from motor's async insert_one() to PyMongo's synchronous insert_one()
        db["NFT"].insert_one({
            "name": image_name,
            "imageBase64": image_base64,
            "embedding": image_embedding.tolist()
        })

        return {"message": "NFT saved successfully in MongoDB"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
