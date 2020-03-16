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

with open('./data_new3/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)
    for line in reader: 
        #print(line)
        lines.append(line)


        
# Preprocessing and Generator functions         
        
def preprocessing(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)  
    img = img[ 55: 140, :, :]   # cropping
    img = cv2.resize(img, (200, 66), interpolation = cv2.INTER_AREA)   # resizing
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)  #chaning color space
    #img = np.expand_dims(img, axis=2)
    return img
 
def augmentation(img):
  
    # color swap
    #random_multiplier = int(np.random.uniform()) * 3 # 0 or 1 or 2
    img_aug = img
    #for i in range(3):
    #    img_aug[i] = img[(i + random_multiplier) % 3]
    
    # color flip 
    #rm = int(np.random.uniform())   # 0 or 1
    #img_aug = rm * img_aug + rm * (255 - img_aug)
    
    # random brightness 
    random_bright = 0.5 + np.random.uniform()
    img_aug[:,:,2] = img_aug[:,:,2]*random_bright
    img_aug[:,:,2][img_aug[:,:,2]>255]  = 255
    
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
                
                for i in range(3):
                    
                    #loading 
                    source_path = batch_sample[i]
                    #print('source_path:', source_path)
                    image = cv2.imread(source_path)
                    
                    
                    # preprocess and augumentation                  
                    image = augmentation(image)
                    image = preprocessing(image)
                    
                    image_flipped = np.fliplr(image)
                    
                     
                    images.append(image)
                    images.append(image_flipped)
                                   
                    # intentional adjustment for right and left cameras
                    bias_adj = 0.20
                    ster_adj = 0.0       
                    if i == 1: ster_adj = bias_adj * 1
                    elif i == 2: ster_adj = bias_adj * -1 

                    measurement = float(batch_sample[3]) + ster_adj
                    measurement_flipped = -measurement     
                    measurements.append(measurement)                    
                    measurements.append(measurement_flipped)
                    
                    #print(source_path, 'ster_adj:', ster_adj, 'measurement:', measurement)            
                    
            X_train = np.array(images)
            y_train = np.array(measurements)

            yield X_train, y_train


# Make Network 

model = Sequential()
model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(66, 200, 3)))
model.add(Conv2D(24, (5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(36, (5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(48, (5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(64, (3, 3),  padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3),  padding='same', activation='relu'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1000, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
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
                    epochs=2, \
                    verbose=1)


model.save('model.h5') 



    