# %%

#import modules

from random import Random
import geopandas as gpd
import pandas as pd
from rasterio.mask import mask
import rasterio as rio
from shapely.geometry import mapping
import matplotlib.pyplot as plt
from rasterio.plot import show
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import time
# %%

#set paths

dir = r'/Users/melindafemminis/Documents/UNIL/UNIL_SA22/machine_learning_env_Sc/sentinel_classification'
poly_shapefile = r'shp/poly_data.shp'
sentAfter = r'sentinel_data/sentinel2-allbands-after.tif'
sentBefore = r'sentinel_data/sentinel2-allbands-before.tif'

# %% 

#open raster file, check infos and see how small we can make it to optimize time

rasterAfter = rio.open(sentAfter)
print("Raster Sentinel2 from after the collapse has %d bands that are %s, %d columns, %d rows and its CRS is %s." % (rasterAfter.count, rasterAfter.descriptions, rasterAfter.width, rasterAfter.height, rasterAfter.crs))


clipped_img = rasterAfter.read([2,3,4])[:, 150:, 100:4500]
fig, ax = plt.subplots(figsize=(10,7))
show(clipped_img[:, :, :], ax=ax, transform=rasterAfter.transform) # add the transform arg to get it in lat long coords


# %% 

#open shapefile with polygons

poly_gpd = gpd.read_file(poly_shapefile)


# %%

def build_train_data(polygons, raster):

    #extract the geometry of each feature in the shapefile in GeoJSON format
    geoms = polygons.geometry.values
    print('The shapelfile has %d different polygons.' % (len(geoms))) 

    X = np.array([], dtype=np.int8).reshape(0,12) # pixels for training
    y = np.array([], dtype=np.string_) # labels for training

    #extract the raster values within the polygons
    with rio.open(raster) as src:
        band_count = src.count
        for index, geom in enumerate(geoms):
            feature = [mapping(geom)]

            # the mask function returns an array of the raster pixels within this feature
            out_image, out_transform = mask(src, feature, crop=True) 
            # eliminate all the pixels with 0 values for all 8 bands - AKA not actually part of the shapefile
            out_image_trimmed = out_image[:,~np.all(out_image == 0, axis=0)]
            # eliminate all the pixels with 255 values for all 8 bands - AKA not actually part of the shapefile
            out_image_trimmed = out_image_trimmed[:,~np.all(out_image_trimmed == 255, axis=0)]
            # reshape the array to [pixel count, bands]
            out_image_reshaped = out_image_trimmed.reshape(-1, band_count)
            # append the labels to the y array
            y = np.append(y,[polygons["classid"][index]] * out_image_reshaped.shape[0]) 
            # stack the pizels onto the pixel array
            X = np.vstack((X,out_image_reshaped))   

    return X, y    


X, y = build_train_data(poly_gpd, sentAfter)




# %% 

#check X and y

#see how many classes; here 0 = water, 1 = land, 2 = urban and 3 = tailings
labels = np.unique(poly_gpd["classid"])
print('The training data include {n} classes: {classes}\n'.format(n=labels.size, 
                                                                classes=labels))

# We will need a "X" matrix containing our features, and a "y" array containing our labels
print('X matrix is sized: {sz}'.format(sz=X.shape))
print('y array is sized: {sz}'.format(sz=y.shape))





# %% 

#split the dataset in train, test and validation data (0.7, 0.15 and 0.15)

X_train, X_next, y_train, y_next = train_test_split(
    X, y,
    test_size = 0.33,
    random_state = 42)

X_test, X_val, y_test, y_val = train_test_split(
    X_next, y_next,
    test_size = 0.5,
    random_state = 42)

print(X_train.shape)
print(X_test.shape)
print(X_val.shape)




# %%

#define parameters' grid and serach for bests

###atm it takes 1.6hour

#starting time
t0 = time.time()

rfc = RandomForestClassifier()

param_grid = {'n_estimators': [50, 100, 200, 300],
               'max_depth': [4, 6, 10, 20],
               'min_samples_split': [2,5,10],
               'min_samples_leaf': [1,10,25],
               'max_features': ['sqrt', 'log2']}

CV_rfc = GridSearchCV(rfc, param_grid, cv=2, verbose=3)

CV_rfc.fit(X_train, y_train)

#print best parameters
print("The best parameters are %s with a score of %0.2f"
    % (CV_rfc.best_params_, CV_rfc.best_score_))

#stop timer
t1 = time.time()
print(f"{(t1 - t0):.2f}s elapsed")  








# %% 

#let's try to manually choose the best parameters. left, max_depth, min_sample_split and min_sample_leaf to default

rfc_manual = RandomForestClassifier(n_estimators=300, max_features='sqrt')
rfc_manual.fit(X_train, y_train)









# %%

#predictions on test set values with best parameters and get accuracy
y_pred_CV_test = CV_rfc.predict(X_test)
test_acc = accuracy_score(y_test, y_pred_CV_test)
print(f'The accuracy of the model with gridsearch on the test set is {test_acc:.1%}')

#predictions on test set values with manual model and get accuracy
y_pred_manual_test = rfc_manual.predict(X_test)
test_acc = accuracy_score(y_test, y_pred_manual_test)
print(f'The accuracy of the model on the test set is {test_acc:.1%}')


# %%

#predictions on test set values with best parameters and get accuracy
y_pred_CV_val = CV_rfc.predict(X_val)
test_acc = accuracy_score(y_val, y_pred_CV_val)
print(f'The accuracy of the model with gridsearch on the validation set is {test_acc:.1%}')

#predictions on test set values with manual model and get accuracy
y_pred_manual_val = rfc_manual.predict(X_val)
test_acc = accuracy_score(y_val, y_pred_manual_val)
print(f'The accuracy of the model on the validation set is {test_acc:.1%}')



# %% 

#classification report on test set 
truth = pd.Series(y_test, name='Actual')
predicted = pd.Series(y_pred_manual_test, name='Predicted')

print(classification_report(truth, predicted))




# %% 

""" # Calculate the confusion matrix on the test set
conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred_manual_test)

# Print the confusion matrix using Matplotlib
fig, ax = plt.subplots(figsize=(7,7))
ax.matshow(conf_matrix, cmap=plt.cm.Reds, alpha=0.7)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix for the test set', fontsize=18)
plt.show() """


from sklearn.metrics import confusion_matrix
import seaborn as sns

classes = ['water', 'land', 'urban', 'tailings']

cm = confusion_matrix(y_test, y_pred_manual_test)
# Normalise
cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=classes, yticklabels=classes, cmap='OrRd')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show(block=False)







# %%
from rasterio.plot import reshape_as_raster, reshape_as_image
#predicting the classes on sentinel-2 images


def predictions(raster):
    #function that crops raster img to smaller size and return classification+img
    with rio.open(raster) as src:
        img = src.read()[:, 150:, 100:4500]

    # Take our full image and reshape into long 2d array (nrow * ncol, nband) for classification
    reshaped_img = reshape_as_image(img)
    print("Predicting the class of 16'360'000 pixels...")
    class_prediction = rfc_manual.predict(reshaped_img.reshape(-1, 12))
    #class_prediction = class_prediction.reshape(reshaped_img[:, :, 0].shape)
    print('Done.')

    return class_prediction, reshaped_img


#get predictions on entine images
class_pred_before, reshaped_img_before = predictions(sentBefore)
class_pred_after, reshaped_img_after = predictions(sentAfter)







# %% 

# get pixel num of class tailings

tailings_before = np.count_nonzero(class_pred_before == b'3')
tailings_after = np.count_nonzero(class_pred_after == b'3')
diff = tailings_after-tailings_before
print('Pixel count with tailings class before the dam collapse is %d, and is %d after the collapse.' % (tailings_before, tailings_after))
print('This is an extra %d pixels!' % (diff))


# %%



def color_stretch(image, index):
    colors = image[:, :, index].astype(np.float64)
    for b in range(colors.shape[2]):
        colors[:, :, b] = rio.plot.adjust_band(colors[:, :, b])
    return colors
    

#show classified image next to predicted classes for image after (currently not working)

fig, axs = plt.subplots(2,1,figsize=(10,10))

img_stretched = color_stretch(reshaped_img_after, [4, 3, 2])
axs[0].imshow(img_stretched)
axs[1].imshow(class_pred_after, cmap='Accent')

fig.show()
