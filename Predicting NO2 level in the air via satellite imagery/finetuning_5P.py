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
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_squared_error

img_rows, img_cols = 224, 224
batch_size = 32
total_epochs = 120
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_val_loss = float('inf') 

#Dataset Loading
csv_path = "/work/cvcs2024/air_pollution_prediction/Sentinel5P_corrected.csv"
try:
    df = pd.read_csv(csv_path, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(csv_path, encoding='ISO-8859-1')

df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.columns = df.columns.str.strip()


path_col = [col for col in df.columns if 'path' in col.lower()][0]
df[path_col] = df[path_col].apply(
    lambda x: os.path.abspath(str(x).strip().replace(';', '').replace(' ', '_').replace('(', '').replace(')', '').replace(',', '_')) if pd.notnull(x) else x
)


label_col = 'Air Pollution Level'
if label_col not in df.columns:
    raise KeyError(f"La colonna '{label_col}' non è presente nel DataFrame")

labels = df[label_col].values
scaler = StandardScaler()
labels = scaler.fit_transform(labels.reshape(-1, 1)).flatten()


class PollutionDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.valid_indices = self._filter_valid_images()

    def _filter_valid_images(self):
        valid_indices = []
        missing_files, invalid_format_files, other_errors = [], [], []

        for i, img_path in enumerate(self.image_paths):
            original_path = img_path
            img_path = os.path.abspath(str(img_path).strip().replace(';', '').replace(' ', '_').replace('(', '').replace(')', '').replace(',', '_'))

            try:
                if os.path.exists(img_path):
                    Image.open(img_path)  
                    valid_indices.append(i)
                else:
                    missing_files.append((original_path, img_path))
            except UnidentifiedImageError:
                invalid_format_files.append((original_path, img_path))
            except Exception as e:
                other_errors.append((img_path, str(e)))

        
        if missing_files:
            print(f"Avviso: {len(missing_files)} immagini non trovate e saranno saltate.")
            for original, missing_file in missing_files:
                print(f"Percorso immagine non trovato: {original} -> {missing_file}")
        if invalid_format_files:
            print(f"Avviso: {len(invalid_format_files)} immagini con formato non valido.")
        if other_errors:
            print(f"Avviso: {len(other_errors)} errori imprevisti durante il caricamento delle immagini.")
            
        return valid_indices

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
            raise e


def augment_data(transform):
    return transforms.Compose([
        transforms.Resize((img_rows, img_cols)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])


transform = augment_data(None)  


image_paths = df[path_col].values
dataset = PollutionDataset(image_paths, labels, transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)



class ResNetForPollution(nn.Module):
    def __init__(self, base_model, num_classes):
        super(ResNetForPollution, self).__init__()
        self.base_model = nn.Sequential(*list(base_model.children())[:-1])  # Remove last layer
        self.fc = nn.Linear(base_model.fc.in_features, num_classes)
        self.dropout = nn.Dropout(0.5) 

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x) 
        x = self.fc(x)
        return x


num_classes = 1  
base_resnet = models.resnet50(pretrained=True)
model = ResNetForPollution(base_resnet, num_classes).to(device)


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


def find_latest_checkpoint(directory, pattern):
    checkpoints = glob.glob(os.path.join(directory, pattern))
    if checkpoints:
        return max(checkpoints, key=os.path.getctime)
    return None

# Load last Checkpoint
checkpoint_dir = "/work/cvcs2024/air_pollution_prediction/Checkpoints_test"
checkpoint_path = find_latest_checkpoint(checkpoint_dir, "resnet_5P_*.pth")
if checkpoint_path is None:
    checkpoint_path = "/work/cvcs2024/air_pollution_prediction/Checkpoints/cp-fine-epoch-29.pth"

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    print(f"Modello caricato dal checkpoint: {checkpoint_path}")
    
    
    if checkpoint_path == "/work/cvcs2024/air_pollution_prediction/Checkpoints/cp-fine-epoch-29.pth":
        for param in base_resnet.parameters():
            param.requires_grad = False
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    start_epoch = checkpoint['epoch'] + 1 if 'epoch' in checkpoint else 0
else:
    print(f"Checkpoint non trovato: {checkpoint_path}")
    start_epoch = 0

def calculate_rmse(outputs, labels):
    outputs = outputs.squeeze().detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    rmse = np.sqrt(np.mean((outputs - labels) ** 2))
    return rmse

# Save the Checkpoint
def save_checkpoint(epoch, model, optimizer, loss, is_best=False):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss
    }
    checkpoint_filename = f"resnet_5P_epoch-{epoch}.pth"
    torch.save(state, os.path.join(checkpoint_dir, checkpoint_filename))
    
    if is_best:
        best_checkpoint_filename = f"resnet_5P_best_epoch-{epoch}.pth"
        torch.save(state, os.path.join(checkpoint_dir, best_checkpoint_filename))
    
    
    with open("/work/cvcs2024/air_pollution_prediction/last_epoch.txt", "w") as f:
        f.write(f"{epoch}")


def extract_features(loader, model):
    model.eval()
    features = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            outputs = model.base_model(inputs)
            features.append(outputs.cpu().numpy())
    return np.concatenate(features, axis=0)

# Training 
def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs, start_epoch=0):
    writer = SummaryWriter()  
    global best_val_loss

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        running_rmse = 0.0

        print(f"Epoch {epoch + 1}/{num_epochs}")

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device).float()

                
                optimizer.zero_grad()

                
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)

                
                loss.backward()
                optimizer.step()

                
                batch_rmse = calculate_rmse(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_rmse += batch_rmse * inputs.size(0)
                pbar.set_postfix({'loss': loss.item(), 'rmse': batch_rmse})
                pbar.update(1)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_rmse = running_rmse / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, RMSE: {epoch_rmse:.4f}")
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('RMSE/train', epoch_rmse, epoch)

        
        model.eval()
        val_loss = 0.0
        val_rmse = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float()
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item() * inputs.size(0)

               
                batch_rmse = calculate_rmse(outputs, labels)
                val_rmse += batch_rmse * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        val_rmse /= len(val_loader.dataset)
        print(f"Validation Loss: {val_loss:.4f}, Validation RMSE: {val_rmse:.4f}")
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('RMSE/val', val_rmse, epoch)
        model.train()

        
        scheduler.step()

        
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        save_checkpoint(epoch, model, optimizer, epoch_loss, is_best=is_best)

    writer.close()

# Train model
if start_epoch < total_epochs:
    train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, total_epochs, start_epoch=start_epoch)

# Save the final model
torch.save(model.state_dict(), '/work/cvcs2024/air_pollution_prediction/Checkpoints/finetuned_model_final.pth')
