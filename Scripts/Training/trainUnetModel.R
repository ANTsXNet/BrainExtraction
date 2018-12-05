library( ANTsR )
library( ANTsRNet )
library( keras )
library( tensorflow )

keras::backend()$clear_session()

Sys.setenv( "CUDA_VISIBLE_DEVICES" = 1 )

classes <- c( "background", "brain" )
numberOfClassificationLabels <- length( classes )

imageMods <- c( "T1" )
channelSize <- length( imageMods )
batchSize <- 8L
segmentationLabels <- c( 0, 1 )

baseDirectory <- '/home/ntustison/Data/'
scriptsDirectory <- paste0( baseDirectory, '/Scripts/BrainExtraction/' )
source( paste0( scriptsDirectory, 'unetBatchGenerator.R' ) )

templateDirectory <- paste0( baseDirectory, 'Templates/' )
reorientTemplateDirectory <- paste0( templateDirectory, '/Kirby/SymmetricTemplate/' )
reorientTemplate <- antsImageRead( paste0( reorientTemplateDirectory, "S_template3_resampled2.nii.gz" ) )
adniToKirbyXfrmFiles <- list.files( paste0( templateDirectory, "ADNI/" ), "adnixKirby*", full.names = TRUE )

dataDirectories <- c()
dataDirectories <- append( dataDirectories, paste0( baseDirectory, "ADNI/" ) )
dataDirectories <- append( dataDirectories, paste0( baseDirectory, "CorticalThicknessData2014/IXI/ThicknessAnts/" ) )
dataDirectories <- append( dataDirectories, paste0( baseDirectory, "CorticalThicknessData2014/Kirby/ThicknessAnts/" ) )
dataDirectories <- append( dataDirectories, paste0( baseDirectory, "CorticalThicknessData2014/NKI/ThicknessAnts/" ) )
dataDirectories <- append( dataDirectories, paste0( baseDirectory, "CorticalThicknessData2014/Oasis/ThicknessAnts/" ) )

brainImageFiles <- c()
for( i in seq_len( length( dataDirectories ) ) )
  {
  imageFiles <- list.files( path = dataDirectories[i],
    pattern = "*BrainSegmentation0N4.nii.gz", full.names = TRUE, recursive = TRUE )
  brainImageFiles <- append( brainImageFiles, imageFiles )
  }

trainingImageFiles <- list()
trainingSegmentationFiles <- list()
trainingMaskFiles <- list()
trainingTransforms <- list()

missingFiles <- c()


cat( "Loading data...\n" )
pb <- txtProgressBar( min = 0, max = length( brainImageFiles ), style = 3 )

count <- 1
for( i in seq_len( length( brainImageFiles ) ) )
  {
  setTxtProgressBar( pb, i )

  subjectId <- basename( brainImageFiles[i] )
  subjectDirectory <- dirname( brainImageFiles[i] )
  subjectId <- sub( "BrainSegmentation0N4.nii.gz", '', subjectId )

  if( grepl( "LongitudinalThicknessANTsNative", subjectDirectory ) )
    {
    brainImageFile <- paste0( subjectDirectory, "/", subjectId, "0.nii.gz" )
    } else {
    t1Directory <- sub( "ThicknessAnts", 'T1', subjectDirectory )
    t1Files <- list.files( t1Directory, pattern = paste0( subjectId, "*" ), full.names = TRUE )

    brainImageFile <- t1Files[1]
    }
  brainMaskFile <- paste0( subjectDirectory, "/", subjectId, "BrainExtractionMask.nii.gz" )

  fwdtransforms <- c()
  invtransforms <- c()
  reorientTransform <- ''

  if( grepl( "LongitudinalThicknessANTsNative", subjectDirectory ) )
    {
    xfrmPrefix <- paste0( subjectId, "TemplateToSubject" )
    xfrmFiles <- list.files( subjectDirectory, pattern = paste0( xfrmPrefix, "*" ), full.names = TRUE )

    fwdtransforms[1] <- xfrmFiles[1]                    # FALSE
    fwdtransforms[2] <- xfrmFiles[3]                    # FALSE
    fwdtransforms[3] <- adniToKirbyXfrmFiles[2]         # FALSE
    fwdtransforms[4] <- adniToKirbyXfrmFiles[1]         # TRUE

    invtransforms[1] <- adniToKirbyXfrmFiles[1]         # FALSE
    invtransforms[2] <- adniToKirbyXfrmFiles[3]         # FALSE
    invtransforms[3] <- xfrmFiles[2]                    # FALSE
    invtransforms[4] <- xfrmFiles[1]                    # TRUE
    } else {
    xfrmPrefix <- paste0( subjectId, "xKirbyTemplate" )
    xfrmFiles <- list.files( subjectDirectory, pattern = paste0( xfrmPrefix, "*" ), full.names = TRUE )

    fwdtransforms[1] <- xfrmFiles[2]                    # FALSE
    fwdtransforms[2] <- xfrmFiles[1]                    # TRUE

    invtransforms[1] <- xfrmFiles[1]                    # FALSE
    invtransforms[2] <- xfrmFiles[3]                    # FALSE
    }


  missingFile <- FALSE
  for( j in seq_len( length( fwdtransforms ) ) )
    {
    if( !file.exists( invtransforms[j] ) || !file.exists( fwdtransforms[j] ) )
      {
      # stop( paste( "Transform file does not exist.\n" ) )
      missingFile <- TRUE
      }
    }

  if( ! file.exists( brainImageFile ) || ! file.exists( brainMaskFile ) )
    {
    # stop( paste( "Transform file does not exist.\n" ) )
    missingFile <- TRUE
    }

  if( missingFile )
    {
    missingFiles <- append( missingFiles, subjectDirectory )
    } else {
    trainingTransforms[[count]] <- list(
      fwdtransforms = fwdtransforms, invtransforms = invtransforms )

    trainingImageFiles[[count]] <- brainImageFile
    trainingMaskFiles[[count]] <- brainMaskFile
    count <- count + 1
    }
  }
cat( "\n" )


###
#
# Create the Unet model
#

resampledImageSize <- dim( reorientTemplate )

# See this thread:  https://github.com/rstudio/tensorflow/issues/272

# with( tf$device( "/cpu:0" ), {
unetModel <- createUnetModel3D( c( resampledImageSize, channelSize ),
  numberOfOutputs = numberOfClassificationLabels,
  numberOfLayers = 4, numberOfFiltersAtBaseLayer = 8, dropoutRate = 0.0,
  convolutionKernelSize = c( 3, 3, 3 ), deconvolutionKernelSize = c( 2, 2, 2 ),
  weightDecay = 1e-5 )
  # } )
load_model_weights_hdf5( unetModel, paste0( scriptsDirectory, "/brainExtractionWeights.h5" ) )

parallel_unetModel <- unetModel # multi_gpu_model( unetModel, gpus = 4 )

# parallel_unetModel %>% compile( loss = loss_multilabel_dice_coefficient_error,
#   optimizer = optimizer_adam( lr = 0.0001 ),
#   metrics = c( multilabel_dice_coefficient ) )

parallel_unetModel %>% compile( loss = "categorical_crossentropy",
  optimizer = optimizer_adam( lr = 0.0001 ),
  metrics = c( "acc" ) )


###
#
# Set up the training generator
#

# Split trainingData into "training" and "validation" componets for
# training the model.

numberOfData <- length( trainingImageFiles )
sampleIndices <- sample( numberOfData )

validationSplit <- floor( 0.8 * numberOfData )
trainingIndices <- sampleIndices[1:validationSplit]
numberOfTrainingData <- length( trainingIndices )
validationIndices <- sampleIndices[( validationSplit + 1 ):numberOfData]
numberOfValidationData <- length( validationIndices )

###
#
# Run training
#

track <- unetModel %>% fit_generator(
  generator = unetImageBatchGenerator( batchSize = batchSize,
                                       resampledImageSize = resampledImageSize,
                                       segmentationLabels = segmentationLabels,
                                       doRandomHistogramMatching = FALSE,
                                       reorientImage = reorientTemplate,
                                       sourceImageList = trainingImageFiles[trainingIndices],
                                       segmentationList = trainingMaskFiles[trainingIndices],
                                       sourceTransformList = trainingTransforms[trainingIndices],
                                       outputFile = paste0( scriptsDirectory, "trainingData.csv" )
                                     ),
  steps_per_epoch = 32L,
  epochs = 75,
  validation_data = unetImageBatchGenerator( batchSize = batchSize,
                                       resampledImageSize = resampledImageSize,
                                       segmentationLabels = segmentationLabels,
                                       doRandomHistogramMatching = FALSE,
                                       reorientImage = reorientTemplate,
                                       sourceImageList = trainingImageFiles[validationIndices],
                                       segmentationList = trainingMaskFiles[validationIndices],
                                       sourceTransformList = trainingTransforms[validationIndices],
                                       outputFile = paste0( scriptsDirectory, "validationData.csv" )
                                     ),
  validation_steps = 16,
  callbacks = list(
    callback_model_checkpoint( paste0( scriptsDirectory, "/brainExtractionWeights.h5" ),
      monitor = 'val_loss', save_best_only = TRUE, save_weights_only = TRUE,
      verbose = 1, mode = 'auto', period = 1 ),
     callback_reduce_lr_on_plateau( monitor = 'val_loss', factor = 0.5,
       verbose = 1, patience = 10, mode = 'auto' ),
     callback_early_stopping( monitor = 'val_loss', min_delta = 0.001,
       patience = 20 )
  )
)

save_model_weights_hdf5( unetModel, paste0( scriptsDirectory, "/brainExtractionWeights.h5" ) )

