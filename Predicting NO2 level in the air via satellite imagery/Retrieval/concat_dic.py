import numpy as np

file_path1 = "/work/cvcs2024/air_pollution_prediction/Retrieval/feature_vectors_5P.npy" 
file_path2 = "/work/cvcs2024/air_pollution_prediction/Retrieval/feature_vectors_S2.npy" 

dict1 = np.load(file_path1, allow_pickle=True).item()
dict2 = np.load(file_path2, allow_pickle=True).item()

if not isinstance(dict1, dict) or not isinstance(dict2, dict):
    raise ValueError("Uno dei file caricati non è un dizionario.")

combined_dict = {**dict1, **dict2}

sorted_keys = sorted(combined_dict.keys(), key=lambda x: x[0])

sorted_combined_dict = {key: combined_dict[key] for key in sorted_keys}

def map_features(feature_vector):
    return feature_vector  

mapped_combined_dict = {key: (map_features(key[1]), value) for key, value in sorted_combined_dict.items()}

# Salva il dizionario finale in un nuovo file .npy
output_path = "/work/cvcs2024/air_pollution_prediction/Retrieval/combined_feature_vectors.npy"  # Modifica con il percorso dove salvare il dizionario combinato
np.save(output_path, mapped_combined_dict)

print(f"Feature vectors combinati e salvati in {output_path}")
