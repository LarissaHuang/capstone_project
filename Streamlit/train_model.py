# Install required packages


import os
import numpy as np

import tensorflow as tf
from keras.layers import Input, Lambda, GlobalAveragePooling2D, Dense, Dropout, Flatten
from keras.models import Model
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping



# IMAGES TO ARRAYS
def preprocess_images(folder_path, target_size=(224, 224)):
    image_arrays = []
    labels = []
    
    for label_name in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label_name)
        if os.path.isdir(label_path):
            for img_file in os.listdir(label_path):
                img_path = os.path.join(label_path, img_file)
                if img_path.endswith(('.jpg', '.jpeg', '.png')):
                    img = image.load_img(img_path, target_size=target_size)
                    img_array = image.img_to_array(img)
                    img_array = preprocess_input(img_array)
                    image_arrays.append(img_array)
                    labels.append(label_name)
    
    return image_arrays, labels

# Path to the train, valid, and test folders
train_folder = '../train'
valid_folder = '../valid'
test_folder = '../test'

# Process train images
train_images, train_labels = preprocess_images(train_folder)

# Process valid images
valid_images, valid_labels = preprocess_images(valid_folder)

# Process test images
test_images, test_labels = preprocess_images(test_folder)


# DATA AUGMENTATION
height = 224
width = 224
channels = 3

train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # for validation data
)

# Create validation image data generator.
valid_datagen = ImageDataGenerator(
      preprocessing_function=tf.keras.applications.efficientnet.preprocess_input)

# Create validation image data generator.
test_datagen = ImageDataGenerator(
      preprocessing_function=tf.keras.applications.efficientnet.preprocess_input)

# Prepare generators
train_generator = train_datagen.flow(np.array(train_images), batch_size=32)
valid_generator = valid_datagen.flow(np.array(valid_images), batch_size=32, shuffle=False)
test_generator = test_datagen.flow(np.array(test_images), batch_size=32, shuffle=False)

# LOADING PRE-TRAINED MODEL
eff_model = EfficientNetB0(weights='imagenet', pooling='max', input_shape=(224, 224, 3), include_top=False)
    
# FREEZE LAYERS IN BASE MODEL
for layer in eff_model.layers:
    layer.trainable=False

# MODEL CUSTOMIZATION
x = eff_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.45)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.45)(x)
output = Dense(525, activation='softmax')(x)

# MODEL INSTANTIATION
model = Model(inputs=eff_model.input, outputs=output)

# MODEL COMPILE AND TRAIN
# # my class labels are integer indices so I set class_mode='sparse' in my train_generator
# # therefore, I will compile my model with 'sparse_categorical_crossentropy':
model.compile(optimizer='Adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Use an early stopping callback to stop training
# once we no longer have improvements in our validation loss
early_stop = EarlyStopping(monitor='val_loss',
                           patience=2,
                           mode='min',
                           verbose=1)

# # Fit the model on the training data, defining desired batch_size & number of epochs,
# # running validation after each batch
model.fit(train_generator,
          epochs=80,
          validation_data = valid_datagen,
          callbacks=[early_stop])

print(os.getcwd())

model.save("custom_model.h5")