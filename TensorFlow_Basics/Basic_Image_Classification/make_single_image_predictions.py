import train_basic_image_classification_model as train
import make_multiple_image_predictions as mmip

#TensorFlow and tf.keras
import tensorflow as tf
#Helper libraries
import numpy as np
import matplotlib.pyplot as plt

#USING THE TRAINED MODEL TO MAKE SINGLE PREDICTION
# Grab an image from the test dataset.
img = train.test_images[1]

#print(img.shape)

#tf.keras models are optimized to make predictions on a batch, or collection, of examples at once. 
#Accordingly, even though you're using a single image, you need to add it to a list:

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))

print(img.shape)

#Now predict the correct label for this image:
#tf.keras.Model.predict returns a list of lists—one list for each image in the batch of data. Grab the predictions for our (only) image in the batch:
predictions_single = train.probability_model.predict(img)

print(predictions_single)

#plot, graph, and visualize the data for the predictions
mmip.plot_value_array(1, predictions_single[0], train.test_labels)
_ = plt.xticks(range(10), train.class_names, rotation=45)
plt.show()

#making single prediction
np.argmax(predictions_single[0])


