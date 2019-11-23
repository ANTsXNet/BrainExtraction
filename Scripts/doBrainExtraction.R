library( ANTsR )
library( ANTsRNet )
library( keras )

args <- commandArgs( trailingOnly = TRUE )

if( length( args ) != 3 )
  {
  helpMessage <- paste0( "Usage:  Rscript doBrainExtraction.R",
    " inputFile outputFile reorientationTemplate\n" )
  stop( helpMessage )
  } else {
  inputFileName <- args[1]
  outputFileName <- args [2]
  reorientTemplateFileName <- args[3]
  }

classes <- c( "background", "brain" )
numberOfClassificationLabels <- length( classes )

imageMods <- c( "T1" )
channelSize <- length( imageMods )

cat( "Reading reorientation template", reorientTemplateFileName )
startTime <- Sys.time()
reorientTemplate <- antsImageRead( reorientTemplateFileName )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

resampledImageSize <- dim( reorientTemplate )

unetModel <- createUnetModel3D( c( resampledImageSize, channelSize ),
  numberOfOutputs = numberOfClassificationLabels,
  numberOfLayers = 4, numberOfFiltersAtBaseLayer = 8, dropoutRate = 0.0,
  convolutionKernelSize = c( 3, 3, 3 ), deconvolutionKernelSize = c( 2, 2, 2 ),
  weightDecay = 1e-5 )


cat( "Loading weights file" )
startTime <- Sys.time()
weightsFileName <- paste0( getwd(), "/brainExtractionWeights.h5" )
if( ! file.exists( weightsFileName ) )
  {
  weightsFileName <- getPretrainedNetwork( "brainExtraction", weightsFileName )
  }
unetModel$load_weights( weightsFileName )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

# Process input

startTimeTotal <- Sys.time()

cat( "Reading ", inputFileName )
startTime <- Sys.time()
image <- antsImageRead( inputFileName, dimension = 3 )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

cat( "Normalizing to template" )
startTime <- Sys.time()
centerOfMassTemplate <- getCenterOfMass( reorientTemplate )
centerOfMassImage <- getCenterOfMass( image )
xfrm <- createAntsrTransform( type = "Euler3DTransform",
  center = centerOfMassTemplate,
  translation = centerOfMassImage - centerOfMassTemplate )
warpedImage <- applyAntsrTransformToImage( xfrm, image, reorientTemplate )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

batchX <- array( data = as.array( warpedImage ),
  dim = c( 1, resampledImageSize, channelSize ) )
batchX <- ( batchX - mean( batchX ) ) / sd( batchX )

cat( "Prediction and decoding" )
startTime <- Sys.time()
predictedData <- unetModel %>% predict( batchX, verbose = 0 )
probabilityImagesArray <- decodeUnet( predictedData, reorientTemplate )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( " (elapsed time:", elapsedTime, "seconds)\n" )

cat( "Renormalize to native space" )
startTime <- Sys.time()
probabilityImage <- applyAntsrTransformToImage( invertAntsrTransform( xfrm ),
  probabilityImagesArray[[1]][[2]], image )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

cat( "Writing", outputFileName )
startTime <- Sys.time()
antsImageWrite( probabilityImage, outputFileName )
endTime <- Sys.time()
elapsedTime <- endTime - startTime
cat( "  (elapsed time:", elapsedTime, "seconds)\n" )

endTimeTotal <- Sys.time()
elapsedTimeTotal <- endTimeTotal - startTimeTotal
cat( "\nTotal elapsed time:", elapsedTimeTotal, "seconds\n\n" )
