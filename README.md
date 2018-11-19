# App:  Brain Extraction

Deep learning app made for T1-weighted MRI brain extraction using ANTsRNet

## Sample usage

```
#
#  Usage:
#    Rscript doBrainExtraction.R inputImage outputImage reorientationTemplate weights
#

$ Rscript doBrainExtraction.R ../Data/Example/1097782_defaced_MPRAGE.nii.gz ./test.nii.gz ../Data/Template/S_template3_resampled.nii.gz ../Data/Weights/brainExtractionWeights.h5

Reading reorientation template ../Data/Template/S_template3_resampled.nii.gz  (elapsed time: 0.02490115 seconds)
Using TensorFlow backend.
Loading weights file ../Data/Weights/brainExtractionWeights.h52018-11-19 10:22:36.758835: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
  (elapsed time: 0.3501251 seconds)
Reading  ../Data/Example/1097782_defaced_MPRAGE.nii.gz  (elapsed time: 0.2626131 seconds)
Normalizing to template  (elapsed time: 0.2282929 seconds)
Prediction and decoding (elapsed time: 30.38575 seconds)
Renormalize to native space  (elapsed time: 0.389267 seconds)
Writing ./test.nii.gz  (elapsed time: 0.4968319 seconds)

Total elapsed time: 31.78905 seconds
```
