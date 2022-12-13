////////////////////////////////////////////
// Load images before and after 
////////////////////////////////////////////

// Load a sentinel and landsat 8 SR image collection
var sentinel_collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")

// Create a point object in Jagersfontein
var point = ee.Geometry.Point(25.42388497852072, -29.77068457017339);

// Define start and end dates
var start_before  = ee.Date('2022-07-11');
var finish_before = ee.Date('2022-09-10');
var start_after  = ee.Date('2022-09-12');
var finish_after = ee.Date('2022-10-11');

// Filter sentinel collection using point object, dates and metadata property
var filteredCollectionBeforeS = sentinel_collection
    .filterBounds(point)
    .filterDate(start_before, finish_before)
    .sort('CLOUD_COVER', true);
var filteredCollectionAfterS = sentinel_collection
    .filterBounds(point)
    .filterDate(start_after, finish_after)
    .sort('CLOUD_COVER', true);
    
var imgBeforeSent = filteredCollectionBeforeS.first();
var imgAfterSent = filteredCollectionAfterS.first();

var bands = ['B.+'];
var afterToExport = imgAfterSent.select(bands);
var beforeToExport = imgBeforeSent.select(bands);

Export.image.toDrive({
  image: afterToExport,
  description: 'sentinel2-allbands-after',
  folder: 'eengine',
  region: ROI,
  scale: 10,
  crs: 'EPSG:4326'
});

Export.image.toDrive({
  image: beforeToExport,
  description: 'sentinel2-allbands-before',
  folder: 'eengine',
  region: ROI,
  scale: 10,
  crs: 'EPSG:4326'
});