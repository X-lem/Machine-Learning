# Convolutional Neural Networks (CNN)

# Installing Theano
# conda install theano pygpu
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras


##### Data Preprocessing

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator


##### Initializing the CNN
cl = Sequential()
# Convolution
cl.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Pooling
cl.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening
cl.add(Flatten())

# Hidden / fully connected layer
cl.add(Dense(output_dim = 128, activation = 'relu'))
# Output layer
cl.add(Dense(output_dim = 1, activation = 'sigmoid'))


# Compiling the CNN
cl.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Fitting the CNN to the images
train_datagen = ImageDataGenerator(rescale = 1./255,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale=1./255)

# Training Set
trs = train_datagen.flow_from_directory('CNN Dataset/training_set',
                                        target_size = (64, 64),
                                        batch_size = 32,
                                        class_mode = 'binary')

# Test set
tes = test_datagen.flow_from_directory('CNN Dataset/test_set',
                                        target_size = (64, 64),
                                        batch_size = 32,
                                        class_mode = 'binary')

cl.fit_generator(trs,
                steps_per_epoch = 8000,
                epochs = 25,
                validation_data = tes,
                validation_steps = 2000)




##### Making the predictions

# Predicint the test set results
y_pred = cl.predict(X_test)
y_pred = (y_pred > 0.5) # True if larger than 0.5

# Making the confustion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)