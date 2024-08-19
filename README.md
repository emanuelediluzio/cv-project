#Predizione dell'Inquinamento dell'Aria

## Introduzione
Il nostro progetto è ispirato dall'uso delle immagini satellitari fornite dal programma Copernicus. Dato l'aumento incessante dell'inquinamento a livello globale, il nostro obiettivo è costruire un modello che possa fornire una rappresentazione concreta dei cambiamenti che ci circondano. Vogliamo creare un modello capace di prevedere la qualità dell'aria, rappresentata da specifici parametri di inquinamento, in una determinata area e in un determinato periodo.

## Obiettivi
L'obiettivo principale di questo progetto è prevedere la qualità dell'aria utilizzando i dataset di immagini satellitari dei satelliti Sentinel-2 e Sentinel-5P. Il satellite Sentinel-5P (S5P), parte del programma Copernicus gestito dall'Agenzia Spaziale Europea (ESA), monitora la qualità dell'aria tramite lo strumento TROPOMI (Tropospheric Monitoring Instrument). TROPOMI misura la composizione atmosferica e raccoglie dati su vari inquinanti, tra cui:
- Diossido di azoto (NO₂)
- Ozono Troposferico (O₃)
- Monossido di Carbonio (CO)
- Anidride Solforosa (SO₂)
- Formaldeide (HCHO)
- Aerosol
- Metano (CH₄)

## Pipeline del Progetto
La pipeline del nostro progetto seguirà questi passaggi:

1. **Elaborazione delle Immagini**: Applicazione di correzioni del rumore radiometrico per regolare i dati delle immagini e algoritmi di riduzione del rumore per migliorare la qualità dei dati.

2. **Algoritmo Geometrico**: Correzione delle distorsioni prospettiche nelle immagini satellitari.

3. **Recupero delle Immagini**: Estrazione dei dati atmosferici dalle immagini multispettrali, concentrandosi sulle concentrazioni di inquinanti come NO₂, SO₂ e particolato.

4. **Componente di Deep Learning**: Utilizzo di una rete neurale convoluzionale (CNN) esistente, come ResNet, migliorandola per l'obiettivo specifico del nostro modello.

## Metriche di Valutazione
Valuteremo le prestazioni del nostro modello utilizzando metriche quantitative come:
- Mean Squared Error (MSE)
- Valori R-squared
- Accuratezza

Queste metriche verranno utilizzate per confrontare le nostre previsioni con i dati di riferimento delle stazioni di monitoraggio della qualità dell'aria. Inoltre, confronteremo i nostri risultati con modelli come AQNet e altri approcci di machine learning (ML) e deep learning (DL) per la previsione dell'inquinamento dell'aria.

## Riferimenti
Un elenco di riferimenti e articoli che abbiamo consultato per questo progetto:
- [Predicting air quality via multimodal AI and satellite imagery - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0031320320305086)
- [GitHub - arnavbansal1/SatellitePollutionCNN: Sviluppo di un algoritmo innovativo per prevedere i livelli di inquinamento dell'aria con accuratezza all'avanguardia utilizzando deep learning e immagini satellitari di GoogleMaps](https://github.com/arnavbansal1/SatellitePollutionCNN)
- [ML based assessment and prediction of air pollution from satellite images during COVID-19 pandemic | Multimedia Tools and Applications (springer.com)](https://link.springer.com/article/10.1007/s11042-020-10062-7)
- [Sentinel-2 Datasets in Earth Engine | Earth Engine Data Catalog | Google for Developers](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2)
- [Sentinel-5P Datasets in Earth Engine | Earth Engine Data Catalog | Google for Developers](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_OFFL_L3_NO2)
