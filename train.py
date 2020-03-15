import csv 
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Cropping2D, Lambda, Reshape, Dropout
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import optimizers
import tensorflow as tf
from sklearn.model_selection import train_test_split


# Make Data Set

def preprocessing(img):
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    return img

lines = []

with open('./data_new2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)
    for line in reader: 
        #print(line)
        lines.append(line)
        
        
images = []
measurements = []

bias_adj = 0.20
ster_adj = 0.0

for line in lines: 
    
    for i in range(3):
      
        source_path = line[i]
        print('source_path:', source_path)
        image = cv2.imread(source_path)
        
        image = preprocessing(image)

        images.append(image)

        if i == 1: ster_adj = bias_adj * 1
        elif i == 2: ster_adj = bias_adj * -1 
        
        measurement = float(line[3]) + ster_adj
        measurements.append(measurement)

        #flippted images 

        image_flipped = np.fliplr(image)
        
        image_flipped = preprocessing(image_flipped)
        
        measurement_flipped = -measurement

        
        images.append(image_flipped)
        measurements.append(measurement_flipped)

       
X = np.array(images)
y = np.array(measurements)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=42)

def generator(X_data, y_data, batch_size=32):
    num_samples = len(X_data)
    while 1: # Loop forever so the generator never terminates
      
        X_data, y_data = shuffle(X_data, y_data)
    
        for offset in range(0, num_samples, batch_size):
            X_batch_ = X_data[offset:offset+batch_size]
            y_batch_ = y_data[offset:offset+batch_size]

            yield X_batch, y_batch

batch_size=32
train_generator = generator(X_train, y_train, batch_size=batch_size)
validation_generator = generator(X_valid, y_valid, batch_size=batch_size)


# Make Network 

model = Sequential()
model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((80, 20),(0, 0))))
model.add(Conv2D(24, (5, 5), padding='same', activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling2D())
model.add(Conv2D(36, (5, 5), padding='same', activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling2D())
model.add(Conv2D(48, (5, 5), padding='same', activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling2D())
model.add(Conv2D(64, (3, 3),  padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(400, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.summary()


# compile and train the model using the generator function


adm = optimizers.Adam(lr=0.001)
model.compile(loss='mean_squared_error', optimizer=adm)
#model.fit(X_train, y_train, batch_size=32, validation_split=0.2, shuffle=True, nb_epoch=2)
model.fit_generator(train_generator, steps_per_epoch=ceil(len(X_train)/batch_size), \
                    validation_data=validation_generator, \
                    validation_steps=ceil(len(X_valid)/batch_size),\
                    epochs=5, verbose=1)


model.save('model.h5') 



    