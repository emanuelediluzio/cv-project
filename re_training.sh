#!/bin/bash
#SBATCH --job-name=finetuning
#SBATCH --output=finetuning_%j.log
#SBATCH --partition=all_serial
#SBATCH --gres=gpu:1
#SBATCH --account=cvcs2024
#SBATCH --time=23:00:00  # Tempo massimo per ciascun job

# Carica l'ambiente Conda
source ~/.bashrc
conda activate new_pytorch_env

# Percorso al file requirements.txt
REQ_FILE="/work/cvcs2024/air_pollution_prediction/requirements.txt"

# Installa le dipendenze dal file requirements.txt una sola volta
if [ ! -f "/work/cvcs2024/air_pollution_prediction/.dependencies_installed" ]; then
    pip install -r $REQ_FILE
    touch /work/cvcs2024/air_pollution_prediction/.dependencies_installed
fi

# Numero massimo di epoche
MAX_EPOCHS=80

# Loop per rilanciare il job finché non è completato
while true
do
    # Esegui lo script di fine-tuning
    python3 /work/cvcs2024/air_pollution_prediction/finetuning2.py

    # Leggi il numero di epoche completate dal file
    if [ -f "/work/cvcs2024/air_pollution_prediction/checkpoints/completed_epochs.txt" ]; then
        completed_epochs=$(cat /work/cvcs2024/air_pollution_prediction/checkpoints/completed_epochs.txt)
    else
        completed_epochs=0
    fi

    # Controlla se tutte le epoche sono state completate
    if [ $completed_epochs -ge $MAX_EPOCHS ]; then
        echo "Training completato con successo per tutte le epoche ($MAX_EPOCHS)."
        break
    else
        echo "Epoche completate: $completed_epochs. Riprendi il training..."
    fi

    # Aspetta qualche secondo prima di rilanciare il job
    sleep 5
done
