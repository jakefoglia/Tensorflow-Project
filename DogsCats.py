"""
Jake Foglia
EE551 Final Project
Using tensorflow to train neural network to differentiate between photos of dogs and cats

partially followed this series: 
https://www.youtube.com/watch?v=gT4F3HGYXf4
"""
import cv2
import numpy as np
#import matplotlib.pyplot as plt
from random import shuffle
from tqdm import tqdm
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard

TRAIN_DIR = "cats_vs_dogs/train"
TEST_DIR = "cats_vs_dogs/test"
IMG_SIZE = 100
CATEGORIES = ["Dog", "Cat"]

# tensorflow configuration
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

def generate_train_data():
    train_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        try:
            label = label_img(img)
            path = os.path.join(TRAIN_DIR,img)
            img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
            train_data.append([img, label])
        except Exception as e:
            pass
        
    shuffle(train_data)
    return train_data
    
def label_img(img_name):
    word_label = img_name.split('.')[-3]
    if word_label == 'cat': return 1
    elif word_label == 'dog': return 0

def generate_trained_model(conv_layer_sizes, dense_layer_sizes, train_data) :
    #len(num_dense_layers) is the number of internal dense layers, so excluding the final output dense layer
    if(len(conv_layer_sizes) < 1) :
        raise Exception('there mustd be atleast one specified conv layer size')

    # generate unique model name
    MODEL_NAME = f'dogsvscats_conv_{conv_layer_sizes}_dense_{dense_layer_sizes}_{int(time.time())}'
    tensorboard = TensorBoard(log_dir=f'logs/{MODEL_NAME}') 

    x = [] # list of image arrays representing each image in the training data
    y = [] # list of bools representing whether training image is a cat or a dog

    for features, label in train_data:
        x.append(features)
        y.append(label)

    # format and convert to numpy arrays
    x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    x = x/255.0

    y = np.array(y)

    # train model
    model = Sequential()

    model.add( Conv2D(conv_layer_sizes[0], (3,3), input_shape =  x.shape[1:]) )
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    for i in range(len(conv_layer_sizes)-1) :
        model.add( Conv2D(conv_layer_sizes[i + 1], (3,3)) )
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Flatten())

    for i in range(len(dense_layer_sizes)) : 
        model.add(Dense(dense_layer_sizes[i]))
        model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    model.fit(x,y,batch_size=32, epochs=10, validation_split = 0.15, callbacks = [tensorboard])


    # save and return model
    model.save(f'logs/{MODEL_NAME}/{MODEL_NAME}.h5')
    return model

def predict(model, file_id):
    path = os.path.join(TEST_DIR, f'{file_id}.jpg')
    img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))

    x = []
    x.append(img)

    x = np.array(x)
    x = x.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    x = x / 255.0

    return model.predict_on_batch(x)[0][0]
 
def main():
    train_model = True # train new model ?
    if(train_model) :
        train_data = generate_train_data()
        model = generate_trained_model([256, 128, 64], [512], train_data) # this will auto log
        # model = generate_trained_model([128, 64, 32], [512], train_data)    
        # model = generate_trained_model([64, 32], [512], train_data) 
        model.save('model.h5')

    model = tf.keras.models.load_model('model.h5') # use the most recent one

    # or specify a specifc logged model
    # model = tf.keras.models.load_model('logs/dogsvscats_conv_[256, 128, 64]_dense_[512]_1548044075/dogsvscats_conv_[256, 128, 64]_dense_[512]_1548044075.h5')

    # test our model
    while(True) :
        input_data = []
        print("Enter the name of the image in the test directory without .jpg")
        file_id = input() 

        try:
            prediction = predict(model, file_id)
        except Exception as e :
            print('Error reading file. Exiting loop.')
            break
        
        strOut = 'cat' if prediction > .5 else 'dog' 
        print(strOut, " with confidence of ", 100*(prediction) if (prediction > .5) else 100*(1 - prediction)  )
    

if __name__ == '__main__': main()