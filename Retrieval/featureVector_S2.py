import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_rows, img_cols = 224, 224
csv_path = "/work/cvcs2024/air_pollution_prediction/Sentinel2_NO2_EU_unique_corrected.csv"  # Modifica con il percorso del tuo CSV
model_path = "/work/cvcs2024/air_pollution_prediction/Checkpoints_S2/resnet_2_epoch-119.pth"  # Modifica con il percorso del modello ResNet salvato
output_path = "/work/cvcs2024/air_pollution_prediction/Retrieval/feature_vectors_S2.npy"  # Modifica con il percorso dove salvare il dizionario dei feature vector

df = pd.read_csv(csv_path)

df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.columns = df.columns.str.strip()

path_col = [col for col in df.columns if 'path' in col.lower()][0]
df[path_col] = df[path_col].apply(
    lambda x: os.path.abspath(str(x).strip().replace(';', '').replace(' ', '_').replace('(', '').replace(')', '').replace(',', '_')) if pd.notnull(x) else x
)

df = df[df[path_col].notna()]

label_col = 'Air Pollution Level'
if label_col not in df.columns:
    raise KeyError(f"La colonna '{label_col}' non è presente nel DataFrame")

labels = df[label_col].values

image_paths = df[path_col].values[-100:]
labels = labels[-100:]

class ResNetForFeatureExtraction(nn.Module):
    def __init__(self, base_model):
        super(ResNetForFeatureExtraction, self).__init__()
        self.base_model = nn.Sequential(*list(base_model.children())[:-1])  # Rimuovi l'ultimo livello

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        return x

base_resnet = models.resnet50(pretrained=False)
model = ResNetForFeatureExtraction(base_resnet).to(device)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint, strict=False)
model.eval()

transform = transforms.Compose([
    transforms.Resize((img_rows, img_cols)),
    transforms.ToTensor(),
])

def extract_features(image_paths, model):
    features_dict = {}
    with torch.no_grad():
        for idx, img_path in enumerate(image_paths):
            if isinstance(img_path, float): 
                print(f"Percorso immagine non valido per l'indice {idx}: {img_path}")
                continue
            try:
                image = Image.open(img_path).convert('RGB')
                image = transform(image).unsqueeze(0).to(device)  
                features = model(image).cpu().numpy().flatten()
                features_dict[(idx, tuple(features))] = labels[idx]  # Associa l'indice del CSV e il feature vector al valore di inquinamento
            except Exception as e:
                print(f"Errore nel caricamento dell'immagine {img_path}: {e}")
                continue
    return features_dict

# Estrai i feature vector
feature_vectors = extract_features(image_paths, model)

# Salva i feature vector e i ground truth in un file numpy
np.save(output_path, feature_vectors)

print(f"Feature vectors salvati in {output_path}")