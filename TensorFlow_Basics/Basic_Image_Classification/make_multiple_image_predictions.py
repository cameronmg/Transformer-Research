#import training file
import train_basic_image_classification_model as train

#TensorFlow and tf.keras
import tensorflow as tf
#Helper libraries
import numpy as np
import matplotlib.pyplot as plt

#run script to define and train image classification model
exec('train')

#Graph this to look at the full set of 10 class predictions.
#functions to plot the values and images 
def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(train.class_names[predicted_label],
                                100*np.max(predictions_array),
                                train.class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


#With the model trained, you can use it to make predictions about some images. 
#Attach a softmax layer to convert the model's linear outputs—logits—to probabilities, which should be easier to interpret.
predictions = train.probability_model.predict(train.test_images)

#Verify our predictions
#Correct prediction labels are blue and incorrect prediction labels are red. 
#The number gives the percentage (out of 100) for the predicted label.
# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], train.test_labels, train.test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], train.test_labels)
plt.tight_layout()
plt.show()



