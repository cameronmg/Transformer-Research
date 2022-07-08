#TensorFlow and tf.keras
import tensorflow as tf

#Helper libraries
import numpy as np
import matplotlib.pyplot as plt

#print(tf.__version__)

#This guide uses the Fashion MNIST dataset which contains 70,000 grayscale images in 10 categories.
# The images show individual articles of clothing at low resolution (28 by 28 pixels)Here, 60,000 images are used to train the network and 10,000 images to evaluate how accurately the network learned to classify images.
# You can access the Fashion MNIST directly from TensorFlow. Import and load the Fashion MNIST data directly from TensorFlow:


fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#The images are 28x28 NumPy arrays, with pixel values ranging from 0 to 255.
# The labels are an array of integers, ranging from 0 to 9. These correspond to the class of clothing the image represents:
#Each image is mapped to a single label. Since the class names are not included with the dataset,
# store them here to use later when plotting the images:
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Preprocessing Data
#The data must be preprocessed before training the network. If you inspect the first image in the training set, you will see that the pixel values fall in the range of 0 to 255:
#plt.figure()
#plt.imshow(train_images[0])
#plt.colorbar()
#plt.grid(False)
#plt.show()

#Scale these values to a range of 0 to 1 before feeding them to the neural network model. To do so, divide the values by 255.
# It's important that the training set and the testing set be preprocessed in the same way:

train_images = train_images / 255.0
test_images = test_images / 255.0

#To verify that the data is in the correct format and that you're ready to build and train the network,
# let's display the first 25 images from the training set and display the class name below each image.

#plt.figure(figsize=(10,10))
#for i in range(25):
#    plt.subplot(5,5,i+1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.grid(False)
    #loop through set and show images
#    plt.imshow(train_images[i], cmap=plt.cm.binary)
#    plt.xlabel(class_names[train_labels[i]])
#plt.show()


#The first layer in this network, tf.keras.layers.Flatten, transforms the format of the images from a two-dimensional array (of 28 by 28 pixels) to a one-dimensional array (of 28 * 28 = 784 pixels).
#Think of this layer as unstacking rows of pixels in the image and lining them up. This layer has no parameters to learn; it only reformats the data.
#After the pixels are flattened, the network consists of a sequence of two tf.keras.layers.Dense layers. These are densely connected, or fully connected, neural layers. The first Dense layer has 128 nodes (or neurons).
# The second (and last) layer returns a logits array with length of 10. Each node contains a score that indicates the current image belongs to one of the 10 classes.
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

#Optimizer —This is how the model is updated based on the data it sees and its loss function.
#Loss function —This measures how accurate the model is during training. You want to minimize this function to "steer" the model in the right direction.
#Metrics —Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified.
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


#   ---------------------------------------------------------------------------------------------------------------------------------
#   |  Training the neural network model requires the following steps:                                                              |
#   |    1. Feed the training data to the model. In this example, the training data is in the train_images and train_labels arrays. |
#   |    2.The model learns to associate images and labels.                                                                         |
#   |    3.You ask the model to make predictions about a test set—in this example, the test_images array.                           |
#   |    4. Verify that the predictions match the labels from the test_labels array.                                                |
#   ---------------------------------------------------------------------------------------------------------------------------------

#To start training, call the model.fit method—so called because it "fits" the model to the training data:
model.fit(train_images, train_labels, epochs=10)

#Next, compare how the model performs on the test dataset:
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

#With the model trained, you can use it to make predictions about some images. 
#Attach a softmax layer to convert the model's linear outputs—logits—to probabilities, which should be easier to interpret.
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

#predictions = probability_model.predict(test_images)
