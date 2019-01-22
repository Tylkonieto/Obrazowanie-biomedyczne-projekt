import os
import numpy
import SimpleITK
import matplotlib.pyplot as plt

from helpers import sitk_show, load_image

# labels for white and grey matter

labelWhiteMatter = 1
labelGrayMatter = 2

# read slice 58 of orginal image

imgOriginal = load_image(idx_slice=58)

# show unprocessed image

sitk_show(imgOriginal)

# smooth image

imgSmooth = SimpleITK.CurvatureFlow(image1=imgOriginal,
                                    timeStep=0.125,
                                    numberOfIterations=5)

# select white matter

imgWhiteMatter = SimpleITK.BinaryThreshold(image1=imgSmooth,
                                           lowerThreshold=120,
                                           upperThreshold=180,
                                           insideValue=labelWhiteMatter)

# select gray matter

imgGrayMatter = SimpleITK.BinaryThreshold(image1=imgSmooth,
                                          lowerThreshold=163,
                                          upperThreshold=280,
                                          insideValue=labelGrayMatter)

# put image labels

imgLabels = imgWhiteMatter | imgGrayMatter

# rescale 'imgSmooth' and cast it to an integer type to match that of 'imgWhiteMatter'

imgSmoothInt = SimpleITK.Cast(SimpleITK.RescaleIntensity(
    imgSmooth), imgWhiteMatter.GetPixelID())

# show result

sitk_show(SimpleITK.LabelOverlay(imgSmoothInt, imgLabels))
