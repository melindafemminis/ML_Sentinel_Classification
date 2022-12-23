#%%

#import modules

import geopandas as gpd
import rasterio as rio
from rasterio.plot import show
import os

# %%

#set paths

dir = r'/Users/melindafemminis/Documents/UNIL/UNIL_SA22/machine_learning_env_Sc/sentinel_classification'
pointfile = 'shp/refactored_points.shp'
sentAfter = r'sentinel_data/sentinel2-allbands-after.tif'

# %%

#load point file

df = gpd.read_file(pointfile) #Create geodataframe from the points and check content
coords = [(x,y) for x, y in zip(df.geometry.x, df.geometry.y)]
df.head()

# %%

# load raster file and vizualize data

sentAfter = rio.open(sentAfter)
print("Raster Sentinel2 from after the collapse has %d bands that are %s, %d columns, %d rows and its CRS is %s." % (sentAfter.count, sentAfter.descriptions, sentAfter.width, sentAfter.height, sentAfter.crs))

fig, ax = plt.subplots(figsize=(12,12))
df.plot(ax=ax, color='orangered')
show(sentAfter, ax=ax)
# %%

#extract data from each band for all points

for i in range(sentAfter.count):
    print("Extracting data from band %s..." % (sentAfter.descriptions[i]))
    df[str(sentAfter.descriptions[i])] = [x[i] for x in sentAfter.sample(coords)]
    print('Done.')


df.to_csv(os.path.join(dir, 'data/points_with_data.csv'), mode='a')
print('Done saving data to csv file.')
