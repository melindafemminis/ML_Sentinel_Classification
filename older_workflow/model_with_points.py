# %%

#import modules

import pandas as pd
import rasterio as rio
import numpy as np
import os
# %%

#set paths

dir = r'/Users/melindafemminis/Documents/UNIL/UNIL_SA22/machine_learning_env_Sc/sentinel_classification'
points = r'data/points_with_data.csv'
sentAfter = r'sentinel_data/sentinel2-allbands-after.tif'
sentBefore = r'sentinel_data/sentinel2-allbands-before.tif'
# %%

#load points data and select bands

bands = all
all = ['classId', 'x', 'y', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
rgs = ['classId', 'x', 'y', 'B2', 'B3', 'B4']
#potentially add more options

dfPoints = pd.read_csv(os.path.join(dir, points))

labels = np.unique(dfPoints["className"])
print('The training data include {n} classes: {classes}\n'.format(n=labels.size, 
                                                                classes=labels))

dfPoints = dfPoints[bands]

# %%

def getLatLong(raster): 
    #function to convert raster pixel values to lat long

    with rio.open(raster) as src:
                band1 = src.read(1)
                height = band1.shape[0]
                width = band1.shape[1]
                cols, rows = np.meshgrid(np.arange(width), np.arange(height))
                xs, ys = rio.transform.xy(src.transform, rows, cols)
                lons= np.array(xs)
                lats = np.array(ys)

    #round lons and lats to fit the training data flaot number
    lons = lons.round(6)
    lats = lats.round(6)

    return lons, lats
# %%

afterLons, afterLats = getLatLong(sentAfter)
print(afterLons)
# %%

