# Source: https://medium.com/@wl8380/building-an-image-vector-database-with-resnet50-and-faiss-%EF%B8%8F-c785c13c6c2f

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

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time

uploaded = {}

class ImageVectorDB:
    def __init__(self, watch_folder=None):
        weights = models.ResNet50_Weights.DEFAULT
        self.model = models.resnet50(weights=weights)
        self.model.eval()
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.transform = weights.transforms()
        self.vector_dim = 2048
        
        self.index = faiss.IndexFlatL2(self.vector_dim)
        self.id_to_metadata = {}

        self.watch_folder = watch_folder
        self.uploaded = {}  

        if self.watch_folder:
            self._start_watching_folder()


    def _start_watching_folder(self):
        class NewImageHandler(FileSystemEventHandler):
            def __init__(self, outer_instance):
                self.outer = outer_instance
                super().__init__()

            def on_created(self, event):
                if not event.is_directory and event.src_path.lower().endswith(
                    (".png", ".jpg", ".jpeg")
                ):
                    filename = os.path.basename(event.src_path)
                    print(f"📥 New image detected: {filename}")

                    try:
                        with open(event.src_path, "rb") as f:
                            content = f.read()
                        self.outer.uploaded[filename] = content

                        image = Image.open(io.BytesIO(content))
                        if image.mode != "RGB":
                            image = image.convert("RGB")
                        metadata = {"filename": filename, "original_path": event.src_path}
                        image_id = self.outer.store_image(image, metadata)
                        print(f"✅ Stored image {filename} with id {image_id}")

                    except Exception as e:
                        print(f"Error processing {filename}: {str(e)}")

        event_handler = NewImageHandler(self)
        self.observer = Observer()
        self.observer.schedule(event_handler, self.watch_folder, recursive=False)
        self.observer.start()
        print(f"👀 Watching folder: {self.watch_folder}")

        def stop_watching(self):
            if hasattr(self, "observer"):
                self.observer.stop()
                self.observer.join()
                print("Stopped watching folder.")


    def search_images(self, query_image, k=5):
        query_embedding = self.get_image_embedding(query_image)
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx >= 0 and distance <= 300:  
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
