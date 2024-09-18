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
import torch.nn.functional as F

class PollutionDataset(Dataset):
    def __init__(self, image_paths1, image_paths2, labels, transform=None):
        self.image_paths1 = image_paths1
        self.image_paths2 = image_paths2
        self.labels = labels
        self.transform = transform

        self.valid_indices = []
        self.missing_files = []
        self.invalid_format_files = []

        for i, (img1_path, img2_path) in enumerate(zip(self.image_paths1, self.image_paths2)):
            try:
                img1_path = os.path.abspath(str(img1_path).strip())
                img2_path = os.path.abspath(str(img2_path).strip())
                
                if os.path.exists(img1_path) and os.path.exists(img2_path):
                    Image.open(img1_path)
                    Image.open(img2_path)
                    self.valid_indices.append(i)
                else:
                    self.missing_files.append((img1_path, img2_path))
            except UnidentifiedImageError:
                self.invalid_format_files.append((img1_path, img2_path))

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

class DualResNetForPollution(nn.Module):
    def __init__(self, num_classes=1):
        super(DualResNetForPollution, self).__init__()
        
        self.resnet1 = models.resnet50(pretrained=False)
        self.resnet2 = models.resnet50(pretrained=False)
        
        self.resnet1 = nn.Sequential(*list(self.resnet1.children())[:-1])
        self.resnet2 = nn.Sequential(*list(self.resnet2.children())[:-1])
        
        combined_features = 2048 * 2  
        self.fc1 = nn.Linear(combined_features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3_regression = nn.Linear(256, num_classes)
        self.fc3_classification = nn.Linear(256, 2) 
        self.fc3_area_type = nn.Linear(256, 2)  
        self.dropout = nn.Dropout(0.5)  

    def forward(self, x1, x2):
        x1 = self.resnet1(x1)
        x2 = self.resnet2(x2)
        
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        
        x = torch.cat((x1, x2), dim=1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        
        regression_output = self.fc3_regression(x)
        classification_output = self.fc3_classification(x)
        area_type_output = self.fc3_area_type(x)
        
        return regression_output, classification_output, area_type_output

def save_checkpoint(epoch, model, optimizer, loss, directory="/work/cvcs2024/air_pollution_prediction/DataAugmentation/RandomRotation/Checkpoint"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    checkpoint_path = os.path.join(directory, f"epoch_{epoch}_loss_{loss:.4f}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f"Checkpoint salvato: {checkpoint_path}")

def find_latest_checkpoint(directory, pattern="epoch_*.pth"):
    checkpoints = [f for f in os.listdir(directory) if f.startswith("epoch_") and f.endswith(".pth")]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda f: os.path.getctime(os.path.join(directory, f)))
        return os.path.join(directory, latest_checkpoint)
    return None

def train_model(model, criterion_regression, criterion_classification, optimizer, train_loader, val_loader, num_epochs, start_epoch=0):
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        running_rmse = 0.0
        running_classification_correct = 0
        running_area_type_correct = 0

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for img1, img2, labels in train_loader:
                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device).float()

                optimizer.zero_grad()

                regression_output, classification_output, area_type_output = model(img1, img2)

                loss_regression = criterion_regression(regression_output.squeeze(), labels)

                binary_labels = (labels > 0.5).long() 
                classification_loss = criterion_classification(classification_output, binary_labels)
                
                area_labels = (labels > 1.0).long()  
                area_type_loss = criterion_classification(area_type_output, area_labels)

                loss = loss_regression + classification_loss + area_type_loss
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * img1.size(0)

                batch_rmse = torch.sqrt(F.mse_loss(regression_output.squeeze(), labels))
                running_rmse += batch_rmse.item() * img1.size(0)

                _, classification_preds = torch.max(classification_output, 1)
                running_classification_correct += torch.sum(classification_preds == binary_labels)

                _, area_type_preds = torch.max(area_type_output, 1)
                running_area_type_correct += torch.sum(area_type_preds == area_labels)

                pbar.set_postfix({'loss': loss.item(), 'rmse': batch_rmse.item()})
                pbar.update(1)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_rmse = running_rmse / len(train_loader.dataset)
        epoch_classification_accuracy = running_classification_correct.double() / len(train_loader.dataset)
        epoch_area_type_accuracy = running_area_type_correct.double() / len(train_loader.dataset)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, RMSE: {epoch_rmse:.4f}, "
              f"Classification Accuracy: {epoch_classification_accuracy:.4f}, "
              f"Area Type Accuracy: {epoch_area_type_accuracy:.4f}")

        model.eval()
        val_loss = 0.0
        val_rmse = 0.0
        val_classification_correct = 0
        val_area_type_correct = 0
        with torch.no_grad():
            for img1, img2, labels in val_loader:
                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device).float()
                regression_output, classification_output, area_type_output = model(img1, img2)

                loss_regression = criterion_regression(regression_output.squeeze(), labels)
                binary_labels = (labels > 0.5).long()
                classification_loss = criterion_classification(classification_output, binary_labels)
                area_labels = (labels > 1.0).long()
                area_type_loss = criterion_classification(area_type_output, area_labels)
                loss = loss_regression + classification_loss + area_type_loss

                val_loss += loss.item() * img1.size(0)

                batch_rmse = torch.sqrt(F.mse_loss(regression_output.squeeze(), labels))
                val_rmse += batch_rmse.item() * img1.size(0)

                _, classification_preds = torch.max(classification_output, 1)
                val_classification_correct += torch.sum(classification_preds == binary_labels)

                _, area_type_preds = torch.max(area_type_output, 1)
                val_area_type_correct += torch.sum(area_type_preds == area_labels)

        val_loss /= len(val_loader.dataset)
        val_rmse /= len(val_loader.dataset)
        val_classification_accuracy = val_classification_correct.double() / len(val_loader.dataset)
        val_area_type_accuracy = val_area_type_correct.double() / len(val_loader.dataset)

        print(f"Validation Loss: {val_loss:.4f}, Validation RMSE: {val_rmse:.4f}, "
              f"Validation Classification Accuracy: {val_classification_accuracy:.4f}, "
              f"Validation Area Type Accuracy: {val_area_type_accuracy:.4f}")

        save_checkpoint(epoch + 1, model, optimizer, val_loss)

        model.train()  

img_rows, img_cols = 224, 224
batch_size = 32
total_epochs = 110
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

csv_path_S2 = "/work/cvcs2024/air_pollution_prediction/Sentinel2_NO2_EU_unique_corrected.csv"
csv_path_S5P = "/work/cvcs2024/air_pollution_prediction/Sentinel5P_corrected.csv"

try:
    df_S2 = pd.read_csv(csv_path_S2, encoding='utf-8')
except UnicodeDecodeError:
    df_S2 = pd.read_csv(csv_path_S2, encoding='ISO-8859-1')

try:
    df_S5P = pd.read_csv(csv_path_S5P, encoding='utf-8')
except UnicodeDecodeError:
    df_S5P = pd.read_csv(csv_path_S5P, encoding='ISO-8859-1')

df_S2.columns = df_S2.columns.str.strip()
df_S5P.columns = df_S5P.columns.str.strip()

path_col_S2 = [col for col in df_S2.columns if 'path' in col.lower()][0]
path_col_S5P = [col for col in df_S5P.columns if 'path' in col.lower()][0]

df_S2[path_col_S2] = df_S2[path_col_S2].astype(str).apply(lambda x: x.strip())
df_S5P[path_col_S5P] = df_S5P[path_col_S5P].astype(str).apply(lambda x: x.strip())

label_col = 'Air Pollution Level'
if label_col not in df_S2.columns:
    raise KeyError(f"La colonna '{label_col}' non è presente nel DataFrame S2")

labels = df_S2[label_col].values
scaler = StandardScaler()
labels = scaler.fit_transform(labels.reshape(-1, 1)).flatten()

transform = transforms.Compose([
    transforms.Resize((img_rows, img_cols)),
    transforms.RandomHorizontalFlip(),  # Data augmentation
    transforms.RandomVerticalFlip(),    # Data augmentation
    #transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
    #transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomRotation(degrees=(0,180)),
    transforms.ToTensor(),
])

image_paths_S2 = df_S2[path_col_S2].values
image_paths_S5P = df_S5P[path_col_S5P].values

dataset = PollutionDataset(image_paths_S2, image_paths_S5P, labels, transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = DualResNetForPollution(num_classes=1).to(device)

criterion_regression = nn.MSELoss()  # Loss per la regressione
criterion_classification = nn.CrossEntropyLoss()  # Loss per le classificazioni binarie
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

checkpoint_S2_path = "/work/cvcs2024/air_pollution_prediction/Checkpoints_S2/resnet_2_epoch-119.pth"
checkpoint_S5P_path = "/work/cvcs2024/air_pollution_prediction/Checkpoints_test/resnet_5P_epoch-99.pth"

if os.path.exists(checkpoint_S2_path):
    checkpoint_S2 = torch.load(checkpoint_S2_path)
    model.resnet1.load_state_dict(checkpoint_S2['state_dict'], strict=False)
    print(f"ResNet1 caricato dal checkpoint: {checkpoint_S2_path}")

if os.path.exists(checkpoint_S5P_path):
    checkpoint_S5P = torch.load(checkpoint_S5P_path)
    model.resnet2.load_state_dict(checkpoint_S5P['state_dict'], strict=False)
    print(f"ResNet2 caricato dal checkpoint: {checkpoint_S5P_path}")

checkpoint_dir = "/work/cvcs2024/air_pollution_prediction/DataAugmentation/RandomRotation/Checkpoint"
latest_checkpoint = find_latest_checkpoint(checkpoint_dir)

start_epoch = 0
if latest_checkpoint:
    checkpoint = torch.load(latest_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"Caricato il checkpoint {latest_checkpoint}, ripresa dall'epoca {start_epoch + 1}")

train_model(model, criterion_regression, criterion_classification, optimizer, train_loader, val_loader, total_epochs, start_epoch=start_epoch)

torch.save(model.state_dict(), '/work/cvcs2024/air_pollution_prediction/DataAugmentation/RandomRotation/Checkpoint')
print("Modello finale salvato.")
