# Quantitative estimation of the area affected by the 2022 Jagersfontain tailings dam's collapse

Personnal project for the course Machine Learning for Earth and Environmental Sciences (UNIL, FGSE, Fall 2022)
*Status: draft* 

In this assignment, I show how a unsupervised classification algorithm (Kmeans) applied to multispectral satellite images is used to quantify the area affected by a recent tailings dam’s collapse in South Africa. The first step is to gather relevant data, here two Sentinel-2 images, then classify each pixel in different classes for each image. The classified pixesl from both images are compared to see how much of a difference there is between both images. The results show that an unsupervised algorithm isn’t the optimal option and the analysis will be done again with a supervised algorithm.

## Data
The data consists of 2 Sentinel-2 images, one from July 13 and one from September 16 2022.
They were aquired on Google Earth Engine. 
See "gee-script.js.

## Methodology

The Kmeans algorithm was used for unsupervised classification. The number of clusters is 4. 

## Results

The results show that the algorithm is capable of highlighting the mudslide. The rest of the however is rather noisy. 

## Next steps 
- update code and create functions that will allow processing of multiple images
- compare the clusters from the image before and the image after the dam's collapse to estimate total affected area in sq/m
- use supervised classification algortitm such as random forest to get better results





