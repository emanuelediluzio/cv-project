import os
import time
from datetime import datetime
import pandas as pd
import pyautogui  # Per catturare screenshot
import re
import base64 # Per codificare l'immagine per l'API
import json # Per parsare la risposta JSON dall'API, se necessario
from typing import Optional # Aggiunto per Optional[type]

# Import del client Mistral AI
try:
    from mistralai import Mistral # Corretto import per Mistral
except ImportError as e_import:
    print("ATTENZIONE: Impossibile importare 'Mistral' da 'mistralai'.")
    print(f"Dettagli dell'errore di importazione: {e_import}")
    print("Verifica che la libreria 'mistralai' sia installata correttamente (es. pip install --upgrade mistralai) e sia una versione recente.")
    Mistral = None

try:
    from PIL import Image # Per manipolare le immagini
except ImportError:
    print("Libreria Pillow (PIL) non trovata. Installala con: pip install Pillow")
    exit()

# --- Configurazione Iniziale ---
INTERVALLO_SCREENSHOT = 30 * 60  # Intervallo in secondi (es. 30 minuti)
CARTELLA_SCREENSHOT = "screenshots_acquisiti"
FILE_EXCEL = "dati_estratti.xlsx"
REGIONE_SCHERMO = (0, 18, 697, 774) # Valori forniti dall'utente

# --- Configurazione per API Mistral ---
MISTRAL_API_KEY = "i35xslTehe8x9jKjCVZ7qffNea6oC9ZR" # Fornita dall'utente
# L'endpoint non è più necessario se si usa il client Mistral, ma il NOME DEL MODELLO sì.
MISTRAL_MODEL_NAME = "pixtral-large-latest" # Aggiornato a un modello con capacità di visione

# Commentiamo le configurazioni di Tesseract
# TESSERACT_LANG = 'eng'
# TESSERACT_CONFIG_DEFAULT = '--psm 6'

def crea_cartella_screenshot():
    """Crea la cartella per gli screenshot se non esiste."""
    if not os.path.exists(CARTELLA_SCREENSHOT):
        try:
            os.makedirs(CARTELLA_SCREENSHOT)
            print(f"Cartella '{CARTELLA_SCREENSHOT}' creata con successo.")
        except OSError as e:
            print(f"Errore durante la creazione della cartella '{CARTELLA_SCREENSHOT}': {e}")
            exit()
    else:
        print(f"Cartella '{CARTELLA_SCREENSHOT}' già esistente.")

def cattura_screenshot(percorso_cartella: str, regione: Optional[tuple] = None) -> Optional[str]:
    """
    Cattura uno screenshot e lo salva.
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        nome_file = f"screenshot_{timestamp}.png"
        percorso_completo = os.path.join(percorso_cartella, nome_file)

        if regione:
            print(f"Cattura della regione: {regione}")
            screenshot = pyautogui.screenshot(region=regione)
        else:
            print("ATTENZIONE: REGIONE_SCHERMO non configurata. Cattura dell'intero schermo.")
            screenshot = pyautogui.screenshot()
        
        screenshot.save(percorso_completo)
        print(f"Screenshot salvato in: {percorso_completo}")
        return percorso_completo
    except Exception as e:
        print(f"Errore durante la cattura dello screenshot: {e}")
        return None

def estrai_informazioni_da_immagine(percorso_immagine: str) -> list[dict]:
    print(f"--- Inizio estrazione dati da '{percorso_immagine}' usando API Mistral ---")
    dati_estratti_finali = []

    if MISTRAL_MODEL_NAME == "NOME_MODELLO_MISTRAL_MULTIMODALE_QUI" or not MISTRAL_API_KEY:
        print("ERRORE: MISTRAL_MODEL_NAME non configurato o MISTRAL_API_KEY mancante.")
        print("Per favore, modifica lo script per includere il nome corretto del modello Mistral AI.")
        return []
    
    if Mistral is None:
        print("ERRORE: La libreria 'mistralai' (Mistral) non è caricata correttamente. Vedi messaggi precedenti sull'importazione.")
        return []

    try:
        client = Mistral(api_key=MISTRAL_API_KEY) # Aggiornato inizializzazione

        with open(percorso_immagine, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        image_data_url = f"data:image/png;base64,{encoded_image}"

        prompt_testuale = """Analizza l'immagine di questa chat fornita. L'immagine contiene messaggi di testo. 
Il tuo compito è estrarre le informazioni relative a messaggi che indicano una vincita. 
Cerca pattern come 'Win: [numero] IQ' e l'orario (nel formato HH:MM) associato a quel messaggio. 
Restituisci i risultati come una lista JSON. Ogni elemento della lista dovrebbe essere un oggetto JSON 
con due chiavi: 'valore_iq' (che contiene il numero estratto dopo 'Win:' e prima di 'IQ', come intero) 
e 'orario_messaggio' (che contiene l'orario estratto nel formato stringa 'HH:MM'). 
Se non trovi messaggi rilevanti, restituisci una lista vuota [].
Esempio di output atteso se trovi dati: [{"valore_iq": 1, "orario_messaggio": "18:40"}, {"valore_iq": 3, "orario_messaggio": "18:45"}]
IMPORTANTE: La tua risposta DEVE contenere SOLO la lista JSON e nessun altro testo, commento o spiegazione."""

        # La libreria client di Mistral potrebbe avere un modo più diretto per passare immagini.
        # Questo formato emula quello che si userebbe con una chiamata API diretta.
        
        # Ripristino della struttura corretta per l'invio multimodale
        # come da documentazione più recente di Mistral per modelli con visione.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_testuale},
                    # Assicurati che il tipo MIME (image/png o image/jpeg) sia corretto
                    # per il formato dell'immagine che stai inviando (pyautogui salva in PNG di default)
                    {"type": "image_url", "image_url": image_data_url}
                ]
            }
        ]
        
        # La vecchia struttura che causava l'errore Pydantic con content come lista:
        # messages = [
        #     ChatMessage(role="user", content=[
        #         {"type": "text", "text": prompt_testuale},
        #         {"type": "image_url", "image_url": {"url": image_data_url}}
        #     ])
        # ]

        print(f"Invio richiesta al modello Mistral: {MISTRAL_MODEL_NAME}")
        chat_response = client.chat.complete( # Aggiornata chiamata API
            model=MISTRAL_MODEL_NAME,
            messages=messages
            # Potrebbe essere necessario specificare `max_tokens` o altri parametri qui
        )

        print("--- Risposta ricevuta dall'API Mistral: ---")
        print(chat_response)
        print("-----------------------------------------")

        if chat_response.choices and len(chat_response.choices) > 0:
            messaggio_modello = chat_response.choices[0].message.content
            if messaggio_modello:
                try:
                    lista_risultati = json.loads(messaggio_modello)
                    if isinstance(lista_risultati, list):
                        for item in lista_risultati:
                            if isinstance(item, dict) and 'valore_iq' in item and 'orario_messaggio' in item:
                                dati_estratti_finali.append({
                                    "timestamp_acquisizione_script": datetime.now().isoformat(),
                                    "nome_file_screenshot": os.path.basename(percorso_immagine),
                                    "valore_iq_estratto": int(item['valore_iq']),
                                    "orario_messaggio_estratto": item['orario_messaggio']
                                })
                                print(f"  Trovato da API: IQ={item['valore_iq']}, Orario={item['orario_messaggio']}")
                            else:
                                print(f"Avviso: Elemento non valido nella lista JSON ricevuta: {item}")
                    else:
                        print("Avviso: Il JSON parsato dal modello non è una lista come atteso.")
                except json.JSONDecodeError:
                    print(f"Errore: Impossibile decodificare il messaggio del modello come JSON. Contenuto: \n{messaggio_modello}")
                    print("Verifica che il modello Mistral sia istruito per restituire un JSON valido e che il prompt sia corretto.")
                except Exception as e_parse:
                     print(f"Errore durante il parsing della risposta strutturata: {e_parse}")   
            else:
                print("Nessun contenuto testuale trovato nella risposta del modello.")
        else:
            print("La risposta dell'API non ha la struttura attesa (mancano 'choices').")

    except FileNotFoundError:
        print(f"Errore: File immagine non trovato: {percorso_immagine}")
    except Exception as e:
        # Qui puoi voler loggare più specificamente errori API di Mistral se il client li solleva in modo distinguibile
        print(f"Errore generico durante l'estrazione via API Mistral: {e}")

    if dati_estratti_finali:
        print(f"--- Estrazione API completata. Dati estratti: {len(dati_estratti_finali)} record ---")
    else:
        print("--- Estrazione API completata. Nessun dato rilevante estratto o errore. ---")
        
    return dati_estratti_finali

# Commentiamo la vecchia funzione basata su Tesseract
"""
def estrai_informazioni_da_immagine_tesseract(percorso_immagine: str, tesseract_config: str = TESSERACT_CONFIG_DEFAULT) -> list[dict]:
    # ... (codice tesseract precedente) ...
    pass
"""

def aggiorna_excel(nuovi_dati_lista: list[dict], percorso_excel: str):
    """
    Aggiorna (o crea) un file Excel con i nuovi dati.
    """
    if not nuovi_dati_lista:
        print("Nessun nuovo dato da aggiungere al file Excel.")
        return

    try:
        df_nuovi = pd.DataFrame(nuovi_dati_lista)
        
        if os.path.exists(percorso_excel):
            print(f"Il file Excel '{percorso_excel}' esiste. Provo ad aggiungere i nuovi dati.")
            try:
                # Leggi il file esistente. Se il file è vuoto o malformato, potrebbe dare errore.
                df_esistente = pd.read_excel(percorso_excel)
                if df_esistente.empty and not list(df_nuovi.columns) == list(df_esistente.columns):
                     print("Il file Excel esistente è vuoto e le colonne non corrispondono, lo sovrascrivo.")
                     df_completo = df_nuovi
                else:    
                    df_completo = pd.concat([df_esistente, df_nuovi], ignore_index=True)
                    print(f"Dati esistenti letti. Totale righe prima dell'aggiunta: {len(df_esistente)}")
            except Exception as e_read:
                print(f"Errore durante la lettura del file Excel esistente '{percorso_excel}': {e_read}")
                print("Sovrascrivo il file con i nuovi dati.")
                df_completo = df_nuovi 
        else:
            print(f"Il file Excel '{percorso_excel}' non esiste. Verrà creato.")
            df_completo = df_nuovi

        df_completo.to_excel(percorso_excel, index=False)
        print(f"File Excel '{percorso_excel}' aggiornato/creato con successo. Totale righe: {len(df_completo)}")

    except Exception as e:
        print(f"Errore durante l'aggiornamento del file Excel '{percorso_excel}': {e}")

def main_loop():
    """Ciclo principale per catturare screenshot ed estrarre dati."""
    print("--- Avvio Script Acquisizione Dati da Screenshot (versione API Mistral) ---")
    print(f"Cartella screenshot: '{os.path.abspath(CARTELLA_SCREENSHOT)}'")
    print(f"File Excel dati: '{os.path.abspath(FILE_EXCEL)}'")
    print(f"Intervallo tra acquisizioni: {INTERVALLO_SCREENSHOT / 60} minuti")
    if REGIONE_SCHERMO:
        print(f"Regione di cattura configurata: {REGIONE_SCHERMO}")
    else:
        print("Cattura dell'intero schermo configurata.")
    print(f"API Key Mistral (parziale): {MISTRAL_API_KEY[:4]}...{MISTRAL_API_KEY[-4:]}")
    print(f"Modello Mistral: {MISTRAL_MODEL_NAME}")
    print("----------------------------------------------------")
    
    if Mistral is None and MISTRAL_MODEL_NAME != "NOME_MODELLO_MISTRAL_MULTIMODALE_QUI":
        print("ERRORE CRITICO: La libreria 'mistralai' (Mistral) è necessaria ma non è stata caricata correttamente.")
        print("Vedi messaggi precedenti sull'importazione.")
        exit()
    if MISTRAL_MODEL_NAME == "NOME_MODELLO_MISTRAL_MULTIMODALE_QUI":
        print("AVVISO: Il nome del modello Mistral (MISTRAL_MODEL_NAME) non è configurato. Lo script non potrà estrarre dati via API.")

    crea_cartella_screenshot()

    try:
        while True:
            print(f"\n--- Inizio nuovo ciclo di acquisizione ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")
            
            percorso_screenshot = cattura_screenshot(CARTELLA_SCREENSHOT, REGIONE_SCHERMO)

            if percorso_screenshot:
                print(f"Screenshot catturato: {percorso_screenshot}")
                
                # Chiama la nuova funzione che (tenterà di) usare l'API esterna
                dati_estratti = estrai_informazioni_da_immagine(percorso_screenshot)
                
                if dati_estratti:
                    print(f"Dati estratti (via API Mistral): {len(dati_estratti)} record.")
                    aggiorna_excel(dati_estratti, FILE_EXCEL)
                else:
                    print("Nessun dato rilevante estratto (o errore API Mistral). Vedi log sopra.")
            else:
                print("Cattura screenshot fallita. Salto estrazione e aggiornamento Excel.")

            print(f"Attesa di {INTERVALLO_SCREENSHOT / 60} minuti prima del prossimo ciclo...")
            time.sleep(INTERVALLO_SCREENSHOT)

    except KeyboardInterrupt:
        print("\n--- Interruzione da tastiera ricevuta. Script terminato. ---")
    except Exception as e:
        print(f"\n--- Errore imprevisto nel ciclo principale: {e} ---")
    finally:
        print("--- Uscita dallo script. ---")

if __name__ == "__main__":
    # Assicurati che le librerie necessarie per lo script base siano menzionate
    # pip install pandas openpyxl pyautogui Pillow mistralai
    main_loop() 