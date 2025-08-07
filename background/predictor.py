# Source: 

import torch
from torchvision import models, transforms
from PIL import Image
import faiss
import pickle
import io
import base64
import numpy as np
import os
from pathlib import Path

class ImageVectorDB:
    def __init__(self):
        weights = models.ResNet50_Weights.DEFAULT
        self.model = models.resnet50(weights=weights)
        self.model.eval()
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.transform = weights.transforms()
        self.vector_dim = 2048
        self.index = faiss.IndexFlatL2(self.vector_dim)
        self.id_to_metadata = {}


def upload_images(self, multiple=True):
    uploaded = files.upload()
    stored_images = []
    for filename, content in uploaded.items():
        try:
            image = Image.open(io.BytesIO(content))
            if image.mode != "RGB":
                image = image.convert("RGB")
            metadata = {"filename": filename, "original_path": filename}
            image_id = self.store_image(image, metadata)
            stored_images.append((filename, image_id))
        except Exception as e:
            print(f"Error storing {filename}: {str(e)}")
    return stored_images

def search_images(self, query_image, k=5):
    query_embedding = self.get_image_embedding(query_image)
    distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        if idx >= 0 and distance <= 300:  # Adjust threshold here
            metadata = self.id_to_metadata.get(idx, {}).copy()
            if "image_data" in metadata:
                image_bytes = base64.b64decode(metadata["image_data"])
                metadata["image"] = Image.open(io.BytesIO(image_bytes))
                del metadata["image_data"]
            results.append({"id": idx, "distance": distance, "metadata": metadata})
    return results

def save_database(self, index_file, metadata_file):
    faiss.write_index(self.index, index_file)
    with open(metadata_file, "wb") as f:
        pickle.dump(self.id_to_metadata, f)

def load_database(self, index_file, metadata_file):
    self.index = faiss.read_index(index_file)
    with open(metadata_file, "rb") as f:
        self.id_to_metadata = pickle.load(f)
