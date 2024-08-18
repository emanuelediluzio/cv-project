#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from tqdm import tqdm  # Importa tqdm per la barra di progresso

# Parametri
img_rows, img_cols = 224, 224
batch_size = 32
total_epochs = 30

# Carica il dataset personalizzato
csv_path = "/work/cvcs2024/air_pollution_prediction/EuroSAT_path_category.csv"
df = pd.read_csv(csv_path, sep=';')

# Rimuovi eventuali colonne Unnamed e spazi bianchi dai nomi delle colonne
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.columns = df.columns.str.strip()

# Elimina le righe con valori nulli o vuoti
df = df.dropna(subset=['PATH', 'Category'])

# Stampa i nomi delle colonne per il debug
print("Colonne nel DataFrame:", df.columns)

# Prepara le etichette
if 'Category' in df.columns:
    labels = df['Category'].values
else:
    raise KeyError("La colonna 'Category' non è presente nel DataFrame")

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Dataset personalizzato per caricare le immagini
class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

# Preprocessing e trasformazioni
transform = transforms.Compose([
    transforms.Resize((img_rows, img_cols)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Carica le immagini e crea il dataset
if 'PATH' in df.columns:
    image_paths = df['PATH'].values
else:
    raise KeyError("La colonna 'PATH' non è presente nel DataFrame")

dataset = CustomDataset(image_paths, y, transform=transform)

# Dividi i dati in training e validation set
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Creare DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Carica il modello ResNet50 pre-addestrato su ImageNet
base_model = models.resnet50(pretrained=True)

# Modifica l'ultimo layer del modello per la nostra classificazione
num_features = base_model.fc.in_features
base_model.fc = nn.Sequential(
    nn.Linear(num_features, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, len(label_encoder.classes_)),
    nn.LogSoftmax(dim=1)
)

# Imposta il dispositivo (GPU o CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = base_model.to(device)

# Definisci la funzione di perdita e l'ottimizzatore
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(base_model.parameters(), lr=0.001)

# Percorso per salvare i pesi
checkpoint_path = "/work/cvcs2024/air_pollution_prediction/Checkpoints/cp-fine-epoch-{}.pth"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Funzione per salvare il checkpoint e l'epoca corrente
def save_checkpoint(epoch, model, optimizer, loss):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss
    }
    checkpoint_file = checkpoint_path.format(epoch)
    torch.save(state, checkpoint_file)
    
    if os.path.exists(checkpoint_file):
        with open("/work/cvcs2024/air_pollution_prediction/Checkpoints/last_epoch.txt", "w") as f:
            f.write(str(epoch))

# Funzione per trovare il checkpoint valido più recente
def find_latest_checkpoint():
    checkpoint_file = "/work/cvcs2024/air_pollution_prediction/Checkpoints/last_epoch.txt"
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            epoch = int(f.read().strip())
        checkpoint_path = f"/work/cvcs2024/air_pollution_prediction/Checkpoints/cp-fine-epoch-{epoch}.pth"
        if os.path.exists(checkpoint_path):
            return checkpoint_path, epoch
    return None, 0

# Carica il modello dal checkpoint più recente, se esiste
latest_checkpoint, start_epoch = find_latest_checkpoint()
if latest_checkpoint:
    print(f"Caricamento checkpoint: {latest_checkpoint}")
    checkpoint = torch.load(latest_checkpoint)
    base_model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch += 1
else:
    print("Nessun checkpoint trovato, inizio da zero.")
    start_epoch = 0

# Funzione per calcolare l'accuratezza
def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    corrects = torch.sum(preds == labels.data)
    accuracy = corrects.double() / labels.size(0)
    return accuracy.item()

# Funzione per l'addestramento
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs, start_epoch=0):
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        # Barra di progresso per il training
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")
        
        for inputs, labels in train_loader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)

            # Azzerare i gradienti
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass e ottimizzazione
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(torch.max(outputs, 1)[1] == labels)

            train_loader_tqdm.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        # Salva il checkpoint
        save_checkpoint(epoch + 1, model, optimizer, epoch_loss)

        # Valutazione
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(torch.max(outputs, 1)[1] == labels)
        
        val_loss /= len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
        model.train()

# Addestra il modello
if start_epoch < total_epochs:
    train_model(base_model, criterion, optimizer, train_loader, val_loader, total_epochs, start_epoch=start_epoch)

# Salva il modello finale completo
torch.save(base_model.state_dict(), '/work/cvcs2024/air_pollution_prediction/Checkpoints/fine_tuned_model_final.pth')