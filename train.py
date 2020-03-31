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
import random
import math


# Make Data Set

lines = []

with open('./data1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)
    for line in reader: 
        #print(line)
        lines.append(line)


        
# Preprocessing and Generator functions         
        
def preprocessing(img):
    #img = cv2.GaussianBlur(img, (5, 5), 0)  
    img = img[ 70: 135, :, :]   # cropping
    img = cv2.resize(img, (200, 66), interpolation = cv2.INTER_AREA)   # resizing
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #chaning color space
    #img = np.expand_dims(img, axis=2)
    return img
 
def random_brightness(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)  #chaning color space
    img_aug = img
    
    random_bright = 0.5 + np.random.uniform()
    img_aug[:,:,2] = img_aug[:,:,2]*random_bright
    img_aug[:,:,2][img_aug[:,:,2]>255]  = 255
    
    img_aug = cv2.cvtColor(img_aug, cv2.COLOR_HLS2RGB)
    
    return img_aug
    
def generator(samples, batch_size=32):
    num_samples = len(samples)
             
    while True: # Loop forever so the generator never terminates
      
        random.shuffle(samples)
    
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset : offset+batch_size]
            
            images = []
            measurements = []
            
            for batch_sample in batch_samples:
                
                #loading 
                source_path = batch_sample[0]
                image = cv2.imread(source_path)
                image = preprocessing(image)
                measurement = float(batch_sample[3])
                
                # add original data
                images.append(image)
                measurements.append(measurement)
                
                # augumentation - flip 
                image_flipped = np.fliplr(image)
                images.append(image_flipped)
                measurements.append(measurement * -1)
                
                # augumentation - random brightnes
                image_random_brightness = random_brightness(image)
                images.append(image_random_brightness)
                measurements.append(measurement)
                
                '''
                # blur
                image_blur = cv2.GaussianBlur(image, (5, 5), 0)  
                images.append(image_blur)
                measurements.append(measurement)
                '''
                
                # equalize_hist
                image_equ = np.copy(image)
                for channel in range(image_equ.shape[2]):
                    image_equ[:, :, channel] = cv2.equalizeHist(image_equ[:, :, channel]) * 255
                images.append(image_equ)
                measurements.append(measurement)
                
                # left camera and right camera 
                bias_adj = 0.20
                
                source_path_left = batch_sample[1]  # left
                image_left = cv2.imread(source_path_left)
                image_left = preprocessing(image_left)
                measurement = float(batch_sample[3]) + bias_adj  
                images.append(image_left)
                measurements.append(min(measurement, 1))  
                
                source_path_right = batch_sample[2]  
                image_right = cv2.imread(source_path_right)
                image_right = preprocessing(image_right)
                measurement = float(batch_sample[3]) - bias_adj
                images.append(image_right)
                measurements.append(max(measurement, -1))  
                   
                    
            X_train = np.array(images)
            y_train = np.array(measurements)

            yield X_train, y_train


# Make Network 

model = Sequential()
model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(66, 200, 3)))
model.add(Conv2D(24, (5, 5), padding='same', activation='elu'))
model.add(MaxPooling2D())
model.add(Conv2D(36, (5, 5), padding='same', activation='elu'))
model.add(MaxPooling2D())
model.add(Conv2D(48, (5, 5), padding='same', activation='elu'))
model.add(MaxPooling2D())
model.add(Conv2D(64, (3, 3),  padding='same', activation='elu'))
model.add(Conv2D(64, (3, 3),  padding='same', activation='elu'))
model.add(Flatten())
model.add(Dropout(0.3))
#model.add(Dense(1000, activation='relu'))
model.add(Dense(100, activation='elu'))
model.add(Dense(50, activation='elu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='elu'))
model.add(Dense(1))
model.summary()


# compile and train the model using the generator function
                   
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

batch_size=32
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

adm = optimizers.Adam(lr=0.0005)
model.compile(loss='mean_squared_error', optimizer=adm)
model.fit_generator(train_generator, \
                    steps_per_epoch=math.ceil(len(train_samples)/batch_size), \
                    validation_data=validation_generator, \
                    validation_steps=math.ceil(len(validation_samples)/batch_size),\
                    epochs=6, \
                    verbose=1)


model.save('model.h5') 



    