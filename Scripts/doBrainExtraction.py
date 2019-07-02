import os
import sys
import time
import numpy as np
import keras

import ants
import antspynet

args = sys.argv

if len(args) != 4:
    help_message = ("Usage:  python doBrainExtraction.py" +
        " inputFile outputFile reorientationTemplate")
    raise AttributeError(help_message)
else:
    input_file_name = args[1]
    output_file_name = args[2]
    reorient_template_file_name = args[3]

classes = ("background", "brain")
number_of_classification_labels = len(classes)

image_mods = ["T1"]
channel_size = len(image_mods)

print("Reading reorientation template " + reorient_template_file_name)
start_time = time.time()
reorient_template = ants.image_read(reorient_template_file_name)
end_time = time.time()
elapsed_time = end_time - start_time
print("  (elapsed time: ", elapsed_time, " seconds)")

resampled_image_size = reorient_template.shape

unet_model = antspynet.create_unet_model_3d( (*resampled_image_size, channel_size),
  number_of_outputs = number_of_classification_labels,
  number_of_layers = 4, number_of_filters_at_base_layer = 8, dropout_rate = 0.0,
  convolution_kernel_size = (3, 3, 3), deconvolution_kernel_size = (2, 2, 2),
  weight_decay = 1e-5 )

print( "Loading weights file" )
start_time = time.time()
weights_file_name = "./brainExtractionWeights.h5"

if not os.path.exists(weights_file_name):
    weights_file_name = antspynet.get_pretrained_network("brainExtraction", weights_file_name)

unet_model.load_weights(weights_file_name)
end_time = time.time()
elapsed_time = end_time - start_time
print("  (elapsed time: ", elapsed_time, " seconds)")

start_time_total = time.time()

print( "Reading ", input_file_name )
start_time = time.time()
image = ants.image_read(input_file_name)
image = (image - image.min()) / (image.max() - image.min())
end_time = time.time()
elapsed_time = end_time - start_time
print("  (elapsed time: ", elapsed_time, " seconds)")

print( "Normalizing to template" )
start_time = time.time()
center_of_mass_template = ants.get_center_of_mass(reorient_template)
center_of_mass_image = ants.get_center_of_mass(image)
translation = np.asarray(center_of_mass_image) - np.asarray(center_of_mass_template)
xfrm = ants.create_ants_transform(transform_type="Euler3DTransform",
  center=np.asarray(center_of_mass_template),
  translation=translation)
warped_image = ants.apply_ants_transform_to_image(xfrm, image,
  reorient_template)
warped_image = (warped_image - warped_image.mean()) / warped_image.std()

batchX = np.expand_dims(warped_image.numpy(), axis=0)
batchX = np.expand_dims(batchX, axis=-1)
batchX = (batchX - batchX.mean()) / batchX.std()

end_time = time.time()
elapsed_time = end_time - start_time
print("  (elapsed time: ", elapsed_time, " seconds)")


print("Prediction and decoding")
start_time = time.time()
predicted_data = unet_model.predict(batchX, verbose=0)

origin = reorient_template.origin
spacing = reorient_template.spacing
direction = reorient_template.direction

probability_images_array = list()
probability_images_array.append(
   ants.from_numpy(np.squeeze(predicted_data[0, :, :, :, 0]),
     origin=origin, spacing=spacing, direction=direction))
probability_images_array.append(
   ants.from_numpy(np.squeeze(predicted_data[0, :, :, :, 1]),
     origin=origin, spacing=spacing, direction=direction))

probability_images_array[1]
end_time = time.time()
elapsed_time = end_time - start_time
print("  (elapsed time: ", elapsed_time, " seconds)")

print("Renormalize to native space")
start_time = time.time()
probability_image = ants.apply_ants_transform_to_image(
  ants.invert_ants_transform(xfrm), probability_images_array[1],
  image)
end_time = time.time()
elapsed_time = end_time - start_time
print("  (elapsed time: ", elapsed_time, " seconds)")

print("Writing", output_file_name)
start_time = time.time()
ants.image_write(probability_image, output_file_name)
end_time = time.time()
elapsed_time = end_time - start_time
print("  (elapsed time: ", elapsed_time, " seconds)")

end_time_total = time.time()
elapsed_time_total = end_time_total - start_time_total
print( "Total elapsed time: ", elapsed_time_total, "seconds" )