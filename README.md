# Predicting NO2 Level in the Air via Satellite Imagery

**Alessio Tofoni, Emanuele Di Luzio, Rocco Meoli**  
Dipartimento di Ingegneria "Enzo Ferrari", Università degli Studi di Modena e Reggio Emilia  
Via P. Vivarelli 10, 41125 Modena, Italia  
{337543, 273291, 273824}@studenti.unimore.it  
September 18, 2024

## Abstract

In this study, we propose an advanced pipeline for predicting large-scale atmospheric pollution levels using multispectral satellite images acquired by the Sentinel-2 and Sentinel-5P satellites. The model we developed is based on a combination of two pre-trained and then fine-tuned convolutional neural networks (CNNs), specifically on a ResNet50 architecture. The approach leverages advanced data augmentation techniques and multispectral inputs to provide accurate estimates of atmospheric pollutant concentrations of NO2 over a collection of European cities. This work details the pipeline used, the developed deep learning model, and its validation.

## 1. Introduction

Air pollution constitutes one of the main threats to global health, causing millions of deaths each year and contributing to severe phenomena such as global warming and ocean acidification. Major air pollutants, such as nitrogen dioxide (NO2), carbon monoxide (CO), sulfur dioxide (SO2), and particulate matter (PM), have devastating effects on both the environment and humans.

Traditionally, air pollution monitoring has been carried out through ground-based monitoring stations, which, however, offer limited spatial coverage and are unable to provide a global and continuous view. In contrast, the use of satellites like Sentinel-2 and Sentinel-5P, part of the European Space Agency’s (ESA) Copernicus [^1] program, represents a breakthrough for real-time global air quality monitoring, providing data with high spatial and spectral resolution.

