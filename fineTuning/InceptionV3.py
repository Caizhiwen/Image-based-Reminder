import matplotlib
matplotlib.use("Agg")

from keras.applications.mobilenet import MobileNet

from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.utils import to_categorical
from keras import backend as K

from utils import LoadImages
from sklearn.model_selection import train_test_split
import numpy as np

import matplotlib.pyplot as plt


def fine_tuning():
	#Load Data
	(data,labels,CategoryMapping)=LoadImages("/nfs/ug/homes-0/y/yangze3/Image-based-Reminder/images/original",224,224)
	# scale the raw pixel intensities to the range [0, 1]
	data = np.array(data, dtype="float") / 255.0
	data-=0.5
	data*= 2.


	labels = np.array(labels)
	np.save('CategoryMapping.npy', CategoryMapping) 
	#Get the Category Numbers
	NumberOfClass=len(CategoryMapping)

	print (NumberOfClass)

	# create the base pre-trained model
	base_model = MobileNet(weights='imagenet', include_top=False,input_shape=(224, 224, 3))
	for layer in base_model.layers:
	    layer.trainable = False

	# add a global spatial average pooling layer
	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	# let's add a fully-connected layer
	x = Dense(1024, activation='relu')(x)
	# and a logistic layer -- let's say we have 200 classes
	predictions = Dense(NumberOfClass, activation='softmax')(x)

	# this is the model we will train
	model = Model(inputs=base_model.input, outputs=predictions)

	# first: train only the top layers (which were randomly initialized)
	# i.e. freeze all convolutional InceptionV3 layers
	for layer in base_model.layers:
	    layer.trainable = False

	# compile the model (should be done *after* setting layers to non-trainable)
	model.compile(optimizer='rmsprop', loss='categorical_crossentropy')


# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
	(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=42)

	# convert the labels from integers to vectors
	trainY = to_categorical(trainY, num_classes=NumberOfClass)
	testY = to_categorical(testY, num_classes=NumberOfClass)
	# train the model on the new data for a few epochs
	#model.fit_generator()

	# at this point, the top layers are well trained and we can start fine-tuning
	# convolutional layers from inception V3. We will freeze the bottom N layers
	# and train the remaining top layers.


	batch_size=32
	nb_epoch=15

	model.summary()

	model.fit(trainX, trainY,
              batch_size=batch_size,
              epochs=nb_epoch,
              shuffle=True,
              verbose=1,
              validation_data=(testX, testY),
              )

	# we chose to train the top 2 inception blocks, i.e. we will freeze
	# the first 249 layers and unfreeze the rest:
	for layer in model.layers[:249]:
	   layer.trainable = False
	for layer in model.layers[249:]:
	   layer.trainable = True

	# we need to recompile the model for these modifications to take effect
	# we use SGD with a low learning rate
	from keras.optimizers import SGD
	model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',metrics=["accuracy"])
	result= model.fit(trainX, trainY,
              batch_size=batch_size,
              epochs=nb_epoch,
              shuffle=True,
              verbose=1,
              validation_data=(testX, testY),
              )
	model.evaluate(testX,testY)
	print("[INFO] serializing network...")
	model.save("DenseNet201.model")
	# we train our model again (this time fine-tuning the top 2 inception blocks
	# alongside the top Dense layers
	#model.fit_generator(...)


	# plot the training loss and accuracy
	plt.style.use("ggplot")
	plt.figure()
	N = nb_epoch
	plt.plot(np.arange(0, N), result.history["loss"], label="train_loss")
	plt.plot(np.arange(0, N), result.history["val_loss"], label="val_loss")
	plt.plot(np.arange(0, N), result.history["acc"], label="train_acc")
	plt.plot(np.arange(0, N), result.history["val_acc"], label="val_acc")
	plt.title("Training Loss and Accuracy on InceptionV3")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig("plot.png")

fine_tuning()
