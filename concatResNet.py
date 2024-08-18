import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import StandardScaler
from PIL import Image, UnidentifiedImageError
import pandas as pd
import numpy as np
from tqdm import tqdm

# Dataset personalizzato per caricare le immagini
class PollutionDataset(Dataset):
    def __init__(self, image_paths1, image_paths2, labels, transform=None):
        self.image_paths1 = image_paths1
        self.image_paths2 = image_paths2
        self.labels = labels
        self.transform = transform

        # Filtra i percorsi validi e logga eventuali errori
        self.valid_indices = []
        self.missing_files = []
        self.invalid_format_files = []

        for i, (img1_path, img2_path) in enumerate(zip(self.image_paths1, self.image_paths2)):
            try:
                img1_path = os.path.abspath(str(img1_path).strip())
                img2_path = os.path.abspath(str(img2_path).strip())
                
                if os.path.exists(img1_path) and os.path.exists(img2_path):
                    # Tenta di aprire le immagini per verificare se sono valide
                    Image.open(img1_path)
                    Image.open(img2_path)
                    self.valid_indices.append(i)
                else:
                    self.missing_files.append((img1_path, img2_path))
            except UnidentifiedImageError:
                self.invalid_format_files.append((img1_path, img2_path))

        # Log degli errori trovati
        if self.missing_files:
            print(f"Avviso: {len(self.missing_files)} coppie di immagini non trovate e saranno saltate.")
            for missing_file in self.missing_files:
                print(f"Percorsi delle immagini non trovati: {missing_file}")
        if self.invalid_format_files:
            print(f"Avviso: {len(self.invalid_format_files)} immagini con formato non valido.")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        img1_path = self.image_paths1[real_idx].strip()
        img2_path = self.image_paths2[real_idx].strip()

        try:
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
            label = self.labels[real_idx]
            return img1, img2, label
        except Exception as e:
            print(f"Errore nel caricamento delle immagini {img1_path} e {img2_path}: {e}")
            raise e

# Definizione del modello che combina due ResNet50
class DualResNetForPollution(nn.Module):
    def __init__(self, num_classes=1):
        super(DualResNetForPollution, self).__init__()
        
        # Inizializza due ResNet50 pre-addestrate
        self.resnet1 = models.resnet50(pretrained=True)
        self.resnet2 = models.resnet50(pretrained=True)
        
        # Rimuovi l'ultimo layer (fully connected) di entrambe le ResNet
        self.resnet1 = nn.Sequential(*list(self.resnet1.children())[:-1])
        self.resnet2 = nn.Sequential(*list(self.resnet2.children())[:-1])
        
        # Definisci la fully connected layer finale
        # L'input è il doppio dell'output di una singola ResNet
        combined_features = self.resnet1[7].in_features * 2
        self.fc1 = nn.Linear(combined_features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)  # Dropout layer per prevenire overfitting

    def forward(self, x1, x2):
        # Passa gli input attraverso le due ResNet
        x1 = self.resnet1(x1)
        x2 = self.resnet2(x2)
        
        # Flatten (ridurre la dimensione) i risultati
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        
        # Concatenare gli output delle due ResNet
        x = torch.cat((x1, x2), dim=1)
        
        # Passa l'output concatenato attraverso i fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Aggiungi dropout tra i layers
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

# Funzione per l'addestramento del modello
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for img1, img2, labels in train_loader:
                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device).float()

                # Azzerare i gradienti
                optimizer.zero_grad()

                # Forward pass
                outputs = model(img1, img2)
                loss = criterion(outputs.squeeze(), labels)

                # Backward pass e ottimizzazione
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * img1.size(0)
                pbar.set_postfix({'loss': loss.item()})
                pbar.update(1)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        # Valutazione
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img1, img2, labels in val_loader:
                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device).float()
                outputs = model(img1, img2)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item() * img1.size(0)

        val_loss /= len(val_loader.dataset)
        print(f"Validation Loss: {val_loss:.4f}")
        model.train()

# Parametri e configurazione
img_rows, img_cols = 224, 224
batch_size = 32
total_epochs = 50
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Caricamento e pre-elaborazione dei dati
csv_path = "/work/cvcs2024/air_pollution_prediction/Sentinel2_NO2_EU_unique.csv"
df = pd.read_csv(csv_path)

# Identifica le colonne dei percorsi delle immagini
path_col1 = "Image_Path1"  # Sostituisci con il nome della colonna per il primo set di immagini
path_col2 = "Image_Path2"  # Sostituisci con il nome della colonna per il secondo set di immagini
label_col = 'Air Pollution Level'  # Colonna delle etichette

# Pre-elaborazione delle immagini e delle etichette
df[path_col1] = df[path_col1].astype(str).apply(lambda x: x.strip())
df[path_col2] = df[path_col2].astype(str).apply(lambda x: x.strip())
labels = df[label_col].values
scaler = StandardScaler()
labels = scaler.fit_transform(labels.reshape(-1, 1)).flatten()

transform = transforms.Compose([
    transforms.Resize((img_rows, img_cols)),
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Commentato per mantenere i colori originali
])

image_paths1 = df[path_col1].values
image_paths2 = df[path_col2].values
dataset = PollutionDataset(image_paths1, image_paths2, labels, transform=transform)

# Divisione in training e validation set
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Inizializza il modello, la loss function e l'ottimizzatore
model = DualResNetForPollution(num_classes=1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Addestramento del modello
train_model(model, criterion, optimizer, train_loader, val_loader, total_epochs)

# Salva il modello finale completo
torch.save(model.state_dict(), '/work/cvcs2024/air_pollution_prediction/Checkpoints/dual_resnet_finetuned_model_final.pth')
