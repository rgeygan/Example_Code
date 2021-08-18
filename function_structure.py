import numpy as np
from osgeo import gdal, ogr
import scipy
from skimage import exposure
from skimage.segmentation import slic
import time
from sklearn.ensemble import RandomForestClassifier

##Sets of functions to segments and classify an image. The purpose of this script is to show clean function writing
##and conventions. Sets of naked functions is not a typical super structure that is maintainable or adaptable.

def open_image_data(path: str):
    """
    Function that used GDAL to access band data from some path (local or in cloud storage)
    :param path: path to image
    :return: stacked bands as a single array of data
    """

    #open the dataset, default is read only - this could be developed to access image data in cloud storage as well
    image_data = gdal.Open(path)

    #() after RasterCount made it a function but nbands is actually as class variable?? RasterCount vs RasterCount()
    nbands = image_data.RasterCount

    band_data = []

    #loop through bands, take each raster band and hold it in memory as an numpy array, append each to a list
    for i in range(1, nbands+1):
        band = image_data.GetRasterBand(i).ReadAsArray()
        band_data.append(band)

    #turn list of numpy arrays (band data) into single stacked file - band_data is now one variable, numpy array of all image data
    band_data = np.dstack(band_data)

    return band_data

def make_segments(band_data):
    """
    Function to create segments of an image
    :param band_data: from previous function, band data is the stacked image data
    :return: image segments as its own array --> image
    """
    #need to rescale data 0-1
    img = exposure.rescale_intensity(band_data)

    #create segments
    segments = slic(img, n_segments=8000, compactness=0.01)

    # get driver so we can write a geotiff file later
    driverTiff = gdal.GetDriverByName('GTiff')

    #save segments as GeoTiff
    segments_ds = driverTiff.Create('E:/PyCharm/Proj/segments.tif', image_data.RasterXSize, image_data.RasterYSize, bands=1, eType=gdal.GDT_Float32)

    #Assign a geotransform and projection and write segment info to gtiff file
    segments_ds.SetGeoTransform(image_data.GetGeoTransform())
    segments_ds.SetProjection(image_data.GetProjectionRef())
    segments_ds.GetRasterBand(1).WriteArray(segments)
    segments_ds = None

    return segments_ds, segments

#Previous process takes lots of computing power - could speed that up by paralellizing or using cloud computing environment

#define a function to calculate statistics from segments objects containing spectral data
#this function is used with the next
def segment_stats(segment_pixels):
    features = []
    npixels, nbands = segment_pixels.shape
    #loop through each band grabbing stats for each object (group of pixels in a segment)
    for b in range(nbands):
        stats = scipy.stats.describe(segment_pixels[:,b])
        #this is grabbing only the stats we want skipping the second stat value - see scipy.stats.describe documentation
        band_stats = list(stats.minmax) + list(stats)[2:]
        if npixels == 1:
            #in this case the variance = nan, change it to 0.0
            band_stats[3] == 0.0
        features += band_stats
    return features

def get_segment_stats(segments, img):
    """
    Function that grabs segment ids then loops through segements calculating segments statistics using the segment_stats
    function
    :param segments: segments from image segmentation above
    :param img: our rescale image data
    :return: the segments stats and the segment ids
    """
    #get segment IDs
    segment_ids = np.unique(segments)
    #save to a list
    objects = []
    object_ids = []

    #loop through each seg id, find all pixels in img with that id, calculate statistics for those pixels (with user func)
    #then add statistical "feature" to list and add ids to list as well
    for id in segment_ids:
        segment_pixels = img[segments == id]
        #print(id, segment_pixels.shape) #to show that segment_pixels is a 2d array of pixels in each id and all 4 bands
        #is each pixel as a column and each band as a row with the value being that bands image data for that pixel
        object_features = segment_stats(segment_pixels)
        objects.append(object_features)
        object_ids.append(id)

    return objects, object_ids

def classify(path: str, image_data, segments, objects):
    """
    Function that grabs training data (as vector) from external path, rasterizes training data, and uses it as an input to
    a random forest classifier. The image segmentation itself is what gets classified here.
    :param path: poth to td
    :return:
    """
    #read in training data and get the layer
    train_fn = (path)
    train_ds = ogr.Open(train_fn)
    lyr = train_ds.GetLayer()

    #create a new dataset/rasterize the td
    #create a dataset in memory
    driver  = gdal.GetDriverByName('MEM')
    #file name blank since it's in memory
    target_ds = driver.Create('', image_data.RasterXSize, image_data.RasterYSize, 1, gdal.GDT_UInt16)
    target_ds.SetGeoTransform(image_data.GetGeoTransform())
    target_ds.SetProjection(image_data.GetProjectionRef())

    #rasterize training data and make sure that the output is consistent with the class id number
    options = ['ATTRIBUTE=id']
    gdal.RasterizeLayer(target_ds, [1], lyr, options=options)

    #get information to train model

    #gets the land cover type as a 2d array
    #this is 0 - 7, 0 meaning there's no id information since we rasterized a set of points where many pixels will be 0
    ground_truth = target_ds.GetRasterBand(1).ReadAsArray()

    #get the values for each class not including zero (assuming that the first value is the no data value)
    classes = np.unique(ground_truth)[1:]

    #which segments belong to which class (as a dictionary? a set of sets? so {})
    segments_per_class = {}

    #printing number of segs per class was useful because when I split my TD 50/50 only one vector point for the road
    #class was included - not sufficient for training a classifier
    for klass in classes:
        segments_of_class = segments[ground_truth == klass]
        segments_per_class[klass] = set(segments_of_class)

    #can't have same segement represented by multiple lctype of vic versa
    #need to use sets bc intersection is a method of sets
    intersection = set()
    accum = set()

    #loop through segments to check if they are appearing for more than one class
    #unlikely but it may have been wiser to generate the training dataset AFTER the segmentation with the segmentation
    #up in the background to avoid this issue
    #Not sure what I would I have done if this was an issue - redo TD points??
    for class_segments in segments_per_class.values():
        intersection |= accum.intersection(class_segments)
        accum |= class_segments
    assert len(intersection) == 0, "Segment(s) represent multiple classes"

    #create training image - essentially our segments
    train_img = np.copy(segments)
    #need a threshold to give our training segments new values which need to be greater than seg values?
    threshold = train_img.max() + 1

    #giving each class label a value greater than any in the segment image
    #giving pixels in our training image (which is the segmented image) new values that will represent different classes
    for klass in classes:
        class_label = threshold + klass
        for segment_id in segments_per_class[klass]:
            train_img[train_img == segment_id] = class_label

    #zero out all segments in segmented image that don't have any training data and make all pixels in segs that have a training
    #point equal to the class number
    train_img[train_img <= threshold] = 0
    train_img[train_img > threshold] -= threshold

    #lists that can be used for training
    training_objects = []
    training_labels = []

    for klass in classes:
        class_train_object = [v for i, v in enumerate(objects) if segment_ids[i] in segments_per_class[klass]]
        training_labels += [klass] * len(class_train_object)
        training_objects += class_train_object

    #train classifier using objects and predict all segments based on their stats
    classifier = RandomForestClassifier(n_jobs=-1)
    classifier.fit(training_objects, training_labels)
    predicted = classifier.predict(objects)

    #create copy of segments
    clf = np.copy(segments)
    #link the segment id to the predicted values
    for segment_id, klass, in zip(segment_ids, predicted): #creates a dictionary that links segments_ids to predicted vals
        clf[clf == segment_id] = klass

    #masking to show where we have data and no data
    #a 2d array that has the same number of rows and columns
    #summing just sums across all bands and where there's a 0 means there was no data for any band
    #positive vals where we expect data and neg values where we don't
    mask = np.copy(clf)
    mask[mask <= 7.0] = 1.0
    mask[mask > 7.0] = 0.0
    clf = np.multiply(clf, mask)

    #save classified image to file
    clf_ds = driverTiff.Create('E:/PyCharm/Proj/classified.tif', naip_ds.RasterXSize, naip_ds.RasterYSize, bands=1, eType=gdal.GDT_Float32)

    clf_ds.SetGeoTransform(naip_ds.GetGeoTransform())
    clf_ds.SetProjection(naip_ds.GetProjection())
    clf_ds.GetRasterBand(1).SetNoDataValue(-9999.0)
    clf_ds.GetRasterBand(1).WriteArray(clf)
    clf_ds = None

    return clf_ds