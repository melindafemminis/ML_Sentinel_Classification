# Quantitative estimation of the area affected by the 2022 Jagersfontain tailings dam's collapse

Personnal project for the course Machine Learning for Earth and Environmental Sciences (UNIL, FGSE, Fall 2022)

In this assignment, I show how a supervised classification algorithm (Random Forest) applied to multispectral satellite images is used to quantify the area affected by a recent tailings dam’s collapse in South Africa.

## Data
### Satelite images
The data consists of 2 Sentinel-2 images, one from July 13 and one from September 16 2022.
They were aquired on Google Earth Engine. 
See "gee-script.js.
### Training data
The training data was created in Qgis and is located in the **shp** folder. 

## Script
randomforest-classification.py is the main script. 
