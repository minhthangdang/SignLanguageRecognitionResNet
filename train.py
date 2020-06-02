import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from utils_resnet import load_dataset
from model import ResNet18

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Example of pictures
ROWS = 10
fig, axes = plt.subplots(ROWS, ROWS, figsize=(10, 10))
for i in range(ROWS):
	for j in range(ROWS):
		k = np.random.choice(range(X_train_orig.shape[0]))
		axes[i][j].set_axis_off()
		axes[i][j].imshow(X_train_orig[k].reshape((28, 28)))
#plt.show()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Convert training and test labels to one hot matrices
label_binrizer = LabelBinarizer()
Y_train = label_binrizer.fit_transform(Y_train_orig)
Y_test = label_binrizer.fit_transform(Y_test_orig)

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

# train the neural network
model = ResNet18(input_shape = (28, 28, 1), classes = 24)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs = 2, batch_size = 32)

# test the neural network
preds = model.evaluate(X_test, Y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))