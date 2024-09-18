import numpy as np

# Carica il file .npy
file_path = "/work/cvcs2024/air_pollution_prediction/Retrieval/feature_vectors_S2.npy"
feature_vectors = np.load(file_path, allow_pickle=True).item()  # Carica come un dizionario

print(type(feature_vectors))

print(f"Numero di feature vectors: {len(feature_vectors)}")

# Visualizza alcuni elementi del dizionario
for i, (key, value) in enumerate(feature_vectors.items()):
    if i < 5:  # Mostra solo i primi 5 per esempio
        print(f"Index: {key[0]}, Feature Vector: {key[1][:5]}..., Ground Truth: {value}")
