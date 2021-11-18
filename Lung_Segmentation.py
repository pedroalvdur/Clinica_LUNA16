import numpy as np # linear algebra
import pydicom
import os
import scipy.ndimage as ndimage

from skimage import measure, morphology, segmentation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Load the scans in given folder path
def load_scan(path):
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(-(x.ImagePositionPatient[2])))

    return slices

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):

        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)

# Some of the starting Code is taken from ArnavJain, since it's more readable then my own
def generate_markers(image):
    #Creation of the internal Marker
    marker_internal = image < -400
    marker_internal = segmentation.clear_border(marker_internal)
    marker_internal_labels = measure.label(marker_internal)
    areas = [r.area for r in measure.regionprops(marker_internal_labels)]
    areas.sort()
    if len(areas) > 2:
        for region in measure.regionprops(marker_internal_labels):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                       marker_internal_labels[coordinates[0], coordinates[1]] = 0
    marker_internal = marker_internal_labels > 0
    #Creation of the external Marker
    external_a = ndimage.binary_dilation(marker_internal, iterations=10)
    external_b = ndimage.binary_dilation(marker_internal, iterations=55)
    marker_external = external_b ^ external_a
    #Creation of the Watershed Marker matrix
    marker_watershed = np.zeros((512, 512), dtype=np.int)
    marker_watershed += marker_internal * 255
    marker_watershed += marker_external * 128

    return marker_internal, marker_external, marker_watershed

def seperate_lungs(image):
    #Creation of the markers as shown above:
    marker_internal, marker_external, marker_watershed = generate_markers(image)

    #Creation of the Sobel-Gradient
    sobel_filtered_dx = ndimage.sobel(image, 1)
    sobel_filtered_dy = ndimage.sobel(image, 0)
    sobel_gradient = np.hypot(sobel_filtered_dx, sobel_filtered_dy)
    sobel_gradient *= 255.0 / np.max(sobel_gradient)

    #Watershed algorithm
    watershed = morphology.watershed(sobel_gradient, marker_watershed)

    #Reducing the image created by the Watershed algorithm to its outline
    outline = ndimage.morphological_gradient(watershed, size=(3,3))
    outline = outline.astype(bool)

    #Performing Black-Tophat Morphology for reinclusion
    #Creation of the disk-kernel and increasing its size a bit
    blackhat_struct = [[0, 0, 1, 1, 1, 0, 0],
                       [0, 1, 1, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 1, 1, 0],
                       [0, 0, 1, 1, 1, 0, 0]]
    blackhat_struct = ndimage.iterate_structure(blackhat_struct, 8)
    #Perform the Black-Hat
    outline += ndimage.black_tophat(outline, structure=blackhat_struct)

    #Use the internal marker and the Outline that was just created to generate the lungfilter
    lungfilter = np.bitwise_or(marker_internal, outline)
    #Close holes in the lungfilter
    #fill_holes is not used here, since in some slices the heart would be reincluded by accident
    lungfilter = ndimage.morphology.binary_closing(lungfilter, structure=np.ones((5,5)), iterations=3)

    #Apply the lungfilter (note the filtered areas being assigned -2000 HU)
    segmented = np.where(lungfilter == 1, image, -2000*np.ones((512, 512)))

    return segmented

def shaping(image, scan, new_spacing=[3,3,3]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + list(scan[0].PixelSpacing), dtype=np.float32)
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing

def resample(image, scan):
    resampled,_ = shaping(image, scan)
    print('Original Shape: {} \nFirst Resample: {}'.format(image.shape, resampled.shape))
    z = 0
    y = 0
    x = 0
    while resampled.shape[0]!=120 or resampled.shape[1]!=120 or resampled.shape[2]!=120:

        if resampled.shape[0]-120>=5:
            z+=0.2
            resampled,_ = shaping(image, scan, new_spacing=[3+z,3+y,3+x])
            print('Z Axis \nNew Resample: {} '.format(resampled.shape))

        elif 0<resampled.shape[0]-120<=5:
            z+=0.05
            resampled,_ = shaping(image, scan, new_spacing=[3+z,3+y,3+x])
            print('Z Axis \nNew Resample: {} '.format(resampled.shape))

        elif resampled.shape[0]-120<=-5:
            z-=0.1
            resampled,_ = shaping(image, scan, new_spacing=[3+z,3+y,3+x])
            print('Z Axis \nNew Resample: {} '.format(resampled.shape))

        elif -5<=resampled.shape[0]-120<0:
            z-=0.01
            resampled,_ = shaping(image, scan, new_spacing=[3+z,3+y,3+x])
            print('Z Axis \nNew Resample: {} '.format(resampled.shape))

# X Y Axis
        if resampled.shape[1]-120>=5:
            y+=0.2
            x+=0.2
            resampled,_ = shaping(image, scan, new_spacing=[3+z,3+y,3+x])
            print('X Y Axis \nNew Resample: {} '.format(resampled.shape))

        elif 0<resampled.shape[1]-120<=5:
            y+=0.05
            x+=0.05
            resampled,_ = shaping(image, scan, new_spacing=[3+z,3+y,3+x])
            print('X Y Axis \nNew Resample: {} '.format(resampled.shape))

        elif resampled.shape[1]-120<=-5:
            y-=0.1
            x-=0.1
            resampled,_ = shaping(image, scan, new_spacing=[3+z,3+y,3+x])
            print('X Y Axis \nNew Resample: {} '.format(resampled.shape))

        elif -5<=resampled.shape[1]-120<0:
            y-=0.01
            x-=0.01
            resampled,_ = shaping(image, scan, new_spacing=[3+z,3+y,3+x])
            print('X Y Axis \nNew Resample: {} '.format(resampled.shape))


    resampled, new_spacing = shaping(image, scan, new_spacing=[3+z,3+y,3+x])
    print('Final Shape: {} \nFinal Spacing: {}'.format(resampled.shape, new_spacing))

    return resampled


#np.delete(Resampled_Lung,0,2) # np.delete ---> (image, position to be deleted, Axes (Z=0, Y=1, X=2))z

#Duplicar cortes si menor de 60
#Añadir array de ceros si X o Y menor de 60
