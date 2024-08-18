import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import StandardScaler
from PIL import Image, UnidentifiedImageError
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_squared_error

# Parametri
img_rows, img_cols = 224, 224
batch_size = 32
total_epochs = 80
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_val_loss = float('inf')  # Variabile per tracciare la migliore loss di validazione

# Carica il dataset personalizzato con gestione dell'errore di codifica
csv_path = "/work/cvcs2024/air_pollution_prediction/Sentinel2_NO2_EU_unique_corrected.csv"
try:
    df = pd.read_csv(csv_path, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(csv_path, encoding='ISO-8859-1')

# Elimina le colonne inutili
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Rimuovi eventuali spazi bianchi dai nomi delle colonne
df.columns = df.columns.str.strip()

# Identifica la colonna che contiene il path delle immagini
path_col = [col for col in df.columns if 'path' in col.lower()][0]

# Converti i valori della colonna dei percorsi in stringhe, elimina caratteri non validi e normalizza i percorsi
df[path_col] = df[path_col].apply(lambda x: os.path.abspath(str(x).strip().replace(';', '').replace(' ', '_').replace('(', '').replace(')', '').replace(',', '_')) if pd.notnull(x) else x)

# Prepara le etichette (valori di inquinamento)
label_col = 'Air Pollution Level'  # Usa il nome della colonna corretto per l'etichetta
if label_col not in df.columns:
    raise KeyError(f"La colonna '{label_col}' non è presente nel DataFrame")

labels = df[label_col].values
scaler = StandardScaler()
labels = scaler.fit_transform(labels.reshape(-1, 1)).flatten()

# Dataset personalizzato per caricare le immagini
class PollutionDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

        # Filtra i percorsi validi e logga eventuali errori
        self.valid_indices = []
        self.missing_files = []
        self.invalid_format_files = []
        self.other_errors = []

        for i, img_path in enumerate(self.image_paths):
            original_path = img_path  # salva il percorso originale
            img_path = os.path.abspath(str(img_path).strip().replace(';', '').replace(' ', '_').replace('(', '').replace(')', '').replace(',', '_'))

            try:
                if os.path.exists(img_path):
                    # Tenta di aprire l'immagine per verificare se è valida
                    Image.open(img_path)
                    self.valid_indices.append(i)
                else:
                    self.missing_files.append((original_path, img_path))
            except UnidentifiedImageError:
                self.invalid_format_files.append((original_path, img_path))
            except Exception as e:
                self.other_errors.append((img_path, str(e)))

        # Log degli errori trovati
        if self.missing_files:
            print(f"Avviso: {len(self.missing_files)} immagini non trovate e saranno saltate.")
            for original, missing_file in self.missing_files:
                print(f"Percorso immagine non trovato: {original} -> {missing_file}")
        if self.invalid_format_files:
            print(f"Avviso: {len(self.invalid_format_files)} immagini con formato non valido.")
        if self.other_errors:
            print(f"Avviso: {len(self.other_errors)} errori imprevisti durante il caricamento delle immagini.")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        img_path = self.image_paths[real_idx].strip()

        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            label = self.labels[real_idx]
            return image, label
        except Exception as e:
            print(f"Errore nel caricamento dell'immagine {img_path}: {e}")
            raise e  # Propaga l'errore per ulteriori gestioni se necessario

# Preprocessing e trasformazioni con augmentazioni
transform = transforms.Compose([
    transforms.Resize((img_rows, img_cols)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Carica le immagini e crea il dataset
image_paths = df[path_col].values
dataset = PollutionDataset(image_paths, labels, transform=transform)

# Dividi i dati in training e validation set
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Creare DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Definisci la rete convoluzionale ResNet con congelamento parziale
class ResNetForPollution(nn.Module):
    def __init__(self, base_model, num_classes):
        super(ResNetForPollution, self).__init__()
        self.base_model = nn.Sequential(*list(base_model.children())[:-1])  # Rimuovi l'ultimo livello
        self.fc = nn.Linear(base_model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Inizializza la rete e altri parametri
num_classes = 1  # Output per i livelli di inquinamento
base_resnet = models.resnet50(pretrained=True)

# Congela i primi livelli
for param in base_resnet.parameters():
    param.requires_grad = False

model = ResNetForPollution(base_resnet, num_classes).to(device)

# Definisci la funzione di perdita
criterion = nn.MSELoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define the scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Funzione per trovare il checkpoint più recente
def find_latest_checkpoint(directory, pattern):
    checkpoints = glob.glob(os.path.join(directory, pattern))
    if checkpoints:
        return max(checkpoints, key=os.path.getctime)
    return None

# Cerca un checkpoint con prefisso "resnet_1"
checkpoint_path = find_latest_checkpoint("/work/cvcs2024/air_pollution_prediction/Checkpoints", "resnet_1*.pth")
if checkpoint_path is None:
    # Se non esiste, cerca il vecchio checkpoint
    checkpoint_path = "/work/cvcs2024/air_pollution_prediction/Checkpoints/cp-fine-epoch-29.pth"

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'], strict=False)  # Ignora le chiavi mancanti o aggiuntive
    print(f"Modello caricato dal checkpoint: {checkpoint_path}")
    
    # Ricrea l'ottimizzatore e scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Ricrea l'ottimizzatore
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Ricrea lo scheduler
    print("Ottimizzatore e scheduler ricreati, stato dell'ottimizzatore non caricato.")
else:
    print(f"Checkpoint non trovato: {checkpoint_path}")
    start_epoch = 0

# Imposta start_epoch a 0 per ripartire con il conteggio delle epoche
start_epoch = checkpoint['epoch'] + 1 if 'epoch' in checkpoint else 0

# Funzione per calcolare l'RMSE
def calculate_rmse(outputs, labels):
    # Detach the outputs and labels from the computation graph, convert to numpy
    outputs = outputs.squeeze().detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    # Calculate the RMSE
    rmse = np.sqrt(np.mean((outputs - labels) ** 2))
    return rmse

# Funzione per salvare il checkpoint
def save_checkpoint(epoch, model, optimizer, loss, is_best=False):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss
    }
    checkpoint_filename = f"resnet_1_epoch-{epoch}.pth"
    torch.save(state, f"/work/cvcs2024/air_pollution_prediction/Checkpoints/{checkpoint_filename}")
    
    if is_best:
        best_checkpoint_filename = f"resnet_1_best_epoch-{epoch}.pth"
        torch.save(state, f"/work/cvcs2024/air_pollution_prediction/Checkpoints/{best_checkpoint_filename}")
    
    # Salva il progresso corrente su un file di testo
    with open("/work/cvcs2024/air_pollution_prediction/last_epoch.txt", "w") as f:
        f.write(f"{epoch}")

# Funzione per estrarre i feature vector
def extract_features(loader, model):
    model.eval()
    features = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            outputs = model.base_model(inputs)
            features.append(outputs.cpu().numpy())
    return np.concatenate(features, axis=0)

# Funzione per l'addestramento
def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs, start_epoch=0):
    writer = SummaryWriter()  # Per TensorBoard
    global best_val_loss

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        running_rmse = 0.0

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device).float()

                # Azzerare i gradienti
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)

                # Backward pass e ottimizzazione
                loss.backward()
                optimizer.step()

                # Calcola RMSE per ogni batch
                batch_rmse = calculate_rmse(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_rmse += batch_rmse * inputs.size(0)
                pbar.set_postfix({'loss': loss.item(), 'rmse': batch_rmse})
                pbar.update(1)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_rmse = running_rmse / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, RMSE: {epoch_rmse:.4f}")
        writer.add_scalar('Loss/train', epoch_loss, epoch)  # Log su TensorBoard
        writer.add_scalar('RMSE/train', epoch_rmse, epoch)  # Log su TensorBoard

        # Valutazione
        model.eval()
        val_loss = 0.0
        val_rmse = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float()
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item() * inputs.size(0)

                # Calcola RMSE per la validazione
                batch_rmse = calculate_rmse(outputs, labels)
                val_rmse += batch_rmse * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        val_rmse /= len(val_loader.dataset)
        print(f"Validation Loss: {val_loss:.4f}, Validation RMSE: {val_rmse:.4f}")
        writer.add_scalar('Loss/val', val_loss, epoch)  # Log su TensorBoard
        writer.add_scalar('RMSE/val', val_rmse, epoch)  # Log su TensorBoard
        model.train()

        # Aggiorna lo scheduler del learning rate
        scheduler.step()

        # Salva il checkpoint se la validazione è migliorata
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        save_checkpoint(epoch, model, optimizer, epoch_loss, is_best=is_best)

    writer.close()

# Addestra il modello
if start_epoch < total_epochs:
    train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, total_epochs, start_epoch=start_epoch)

# Salva il modello finale completo
torch.save(model.state_dict(), '/work/cvcs2024/air_pollution_prediction/Checkpoints/finetuned_model_final.pth')
