# src/postprocessing.py

import pandas as pd
import torch
from tqdm import tqdm
from src.preprocessing import idx2soil


def predict(model, dataloader, device):
    model.eval()
    predictions = []
    image_ids = []

    with torch.no_grad():
        for images, ids in tqdm(dataloader):
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            predictions.extend(preds)
            image_ids.extend(ids)

    return image_ids, predictions


def save_submission(image_ids, predictions, output_path="submission.csv"):
    decoded_preds = [idx2soil[p] for p in predictions]
    df = pd.DataFrame({
        "image_id": image_ids,
        "soil_type": decoded_preds
    })
    df.to_csv(output_path, index=False)
    print(f"✅ Submission saved to {output_path}")
