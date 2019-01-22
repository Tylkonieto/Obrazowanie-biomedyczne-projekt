import os
import numpy
import SimpleITK
import matplotlib.pyplot as plt
import time


def sitk_show(img, title=None, margin=0.05, dpi=40):
    nda = SimpleITK.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
    extent = (0, nda.shape[1] * spacing[1], nda.shape[0] * spacing[0], 0)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])

    plt.set_cmap("gray")
    ax.imshow(nda, extent=extent, interpolation=None)

    if title:
        plt.title(title)

    plt.show()

# Directory where the DICOM files are being stored (in this
# case the 'MyHead' folder).
pathDicom = "./MyHead/"

# Z slice of the DICOM files to process. In the interest of
# simplicity, segmentation will be limited to a single 2D
# image but all processes are entirely applicable to the 3D image
idxSlice = 58

# int labels to assign to the segmented white and gray matter.
# These need to be different integers but their values themselves
# don't matter
labelWhiteMatter = 1
labelGrayMatter = 2

# Wczytanie obrazu
reader = SimpleITK.ImageSeriesReader()
filenamesDICOM = reader.GetGDCMSeriesFileNames(pathDicom)
reader.SetFileNames(filenamesDICOM)
imgOriginal = reader.Execute()

imgOriginal = imgOriginal[:, :, idxSlice]

# Wyświetlenie obrazu i rozmycie (lepiej się segmentuje)
sitk_show(imgOriginal)

imgSmooth = SimpleITK.CurvatureFlow(image1=imgOriginal,
                                    timeStep=0.125,
                                    numberOfIterations=5)

# blurFilter = SimpleITK.CurvatureFlowImageFilter()
# blurFilter.SetNumberOfIterations(5)
# blurFilter.SetTimeStep(0.125)
# imgSmooth = blurFilter.Execute(imgOriginal)

#sitk_show(imgSmooth)

lstSeeds = [(150, 75), (187, 54)]

imgWhiteMatter = SimpleITK.ConnectedThreshold(image1=imgSmooth,
                                              seedList=lstSeeds,
                                              lower=120,
                                              upper=175,
                                              replaceValue=labelWhiteMatter)

#sitk_show(imgWhiteMatter)

# Rescale 'imgSmooth' and cast it to an integer type to match that of 'imgWhiteMatter'
imgSmoothInt = SimpleITK.Cast(SimpleITK.RescaleIntensity(imgSmooth), imgWhiteMatter.GetPixelID())

# Use 'LabelOverlay' to overlay 'imgSmooth' and 'imgWhiteMatter'
#sitk_show(SimpleITK.LabelOverlay(imgSmoothInt, imgWhiteMatter))

imgWhiteMatterNoHoles = SimpleITK.VotingBinaryHoleFilling(image1=imgWhiteMatter,
                                                          radius=[2]*3,
                                                          majorityThreshold=1,
                                                          backgroundValue=0,
                                                          foregroundValue=labelWhiteMatter)

#sitk_show(SimpleITK.LabelOverlay(imgSmoothInt, imgWhiteMatterNoHoles))

lstSeeds = [(119, 83), (198, 80), (185, 102), (164, 43), (108, 64), (141,111), (157,115)]

t_start = time.clock()
imgGrayMatter = SimpleITK.ConnectedThreshold(image1=imgSmooth,
                                             seedList=lstSeeds,
                                             lower=163,
                                             upper=280,
                                             replaceValue=labelGrayMatter)
t_end = time.clock()
t = t_end - t_start
print(t)

imgGrayMatterNoHoles = SimpleITK.VotingBinaryHoleFilling(image1=imgGrayMatter,
                                                         radius=[2]*3,
                                                         majorityThreshold=1,
                                                         backgroundValue=0,
                                                         foregroundValue=labelGrayMatter)

#sitk_show(SimpleITK.LabelOverlay(imgSmoothInt, imgGrayMatterNoHoles))

imgLabels = imgWhiteMatterNoHoles | imgGrayMatterNoHoles

sitk_show(SimpleITK.LabelOverlay(imgSmoothInt, imgLabels))

imgMask = (imgWhiteMatterNoHoles/labelWhiteMatter) * (imgGrayMatterNoHoles/labelGrayMatter)
print(imgWhiteMatterNoHoles)
print(imgMask)
imgWhiteMatterNoHoles -= imgMask

imgLabels = imgWhiteMatterNoHoles + imgGrayMatterNoHoles

sitk_show(SimpleITK.LabelOverlay(imgSmoothInt, imgLabels))
