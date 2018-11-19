# App:  Brain Extraction

Deep learning app made for T1-weighted MRI brain extraction using ANTsRNet

## Model training notes

* Training data: IXI, NKI, Kirby, and Oasis
* Unet model (see ``Scripts/Training/``).
* Template-based data augmentation
* Lower resolution training (template size = [80, 96, 96])

## Sample prediction usage

```
#
#  Usage:
#    Rscript doBrainExtraction.R inputImage outputImage reorientationTemplate weights
#
#  MacBook Pro 2016 (no GPU)
#

$ Rscript doBrainExtraction.R ../Data/Example/1097782_defaced_MPRAGE.nii.gz ./outputProbabilityMask.nii.gz ../Data/Template/S_template3_resampled.nii.gz ../Data/Weights/brainExtractionWeights.h5

Reading reorientation template ../Data/Template/S_template3_resampled.nii.gz  (elapsed time: 0.02490115 seconds)
Using TensorFlow backend.
Loading weights file ../Data/Weights/brainExtractionWeights.h52 (elapsed time: 0.3501251 seconds)
Reading  ../Data/Example/1097782_defaced_MPRAGE.nii.gz  (elapsed time: 0.2626131 seconds)
Normalizing to template  (elapsed time: 0.2282929 seconds)
Prediction and decoding (elapsed time: 30.38575 seconds)
Renormalize to native space  (elapsed time: 0.389267 seconds)
Writing ./outputProbabilityMask.nii.gz  (elapsed time: 0.4968319 seconds)

Total elapsed time: 31.78905 seconds
```

## Sample results

![Brain extraction results](Documentation/Images/resultsBrainExtraction.png)
