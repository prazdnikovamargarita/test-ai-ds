from zipfile import ZipFile 
import os
import shutil
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import layers, models


DATADIR = './Animal Image Dataset/train'

if os.path.exists(DATADIR):
    print(" 'Animal Image Dataset' has already exists.")
else:
    
    with ZipFile("./Animal Image Dataset.zip", 'r') as zObject: 
    
        # Extracting all the members of the zip  
        # into a specific location. 
        zObject.extractall( 
            path=".") 
Categories = sorted([name for name in os.listdir(DATADIR) if os.path.isdir(os.path.join(DATADIR, name))])


train_data = os.path.join('Animal Image Dataset/train')

val_data = os.path.join('Animal Image Dataset/validation')

print((train_data))
base_dir = './Animal Image Dataset/train'
categories = [name for _, dirs, _ in os.walk(base_dir) for name in dirs]
print(categories)

categories_len = len(categories)
print(categories_len)



# Set paths for training and validation datasets
BASE_DIR = 'Animal Image Dataset'
TRAINING_DIR = os.path.join(BASE_DIR, 'train')
VALIDATION_DIR = os.path.join(BASE_DIR, 'validation')

# Data augmentation for training images
training_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Reserve a portion for validation
)

# Normalize validation images
validation_datagen = ImageDataGenerator(rescale=1./255)

# Image parameters
IMG_SIZE = (150, 150)  # Resize all images to this size
BATCH_SIZE = 32  # Batch size for training

# Load training data
train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Load validation data
validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)



# Define CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Flatten(),

    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(categories_len, activation='softmax')
])

# Display model summary
model.summary()

# Compile the model
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks to improve training
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=35,
    callbacks=[early_stopping, reduce_lr]
)

import matplotlib.pyplot as plt

# Plot training progress
plt.figure(figsize=(12, 5))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')

plt.show()


# Load an image for prediction
path = 'Animal Image Dataset/validation/dog/dogs_00034.jpg'

# Get model input shape
expected_shape = model.input_shape
print(f"Expected model input shape: {expected_shape}")

# Resize image to match model input size
img = load_img(path, target_size=(expected_shape[1], expected_shape[2]))

# Convert to array and normalize
x = img_to_array(img) / 255.0
x = np.expand_dims(x, axis=0)  # Add batch dimension

# Make a prediction
probabilities = model.predict(x)
predicted_index = np.argmax(probabilities)
predicted_category = Categories[predicted_index]
confidence = probabilities[0][predicted_index] * 100

# Display prediction results
print(f"Image Path: {path}")
print(f"Predicted Category: {predicted_category} with {confidence:.2f}% confidence.")

# Save model weights
model.save('./animal_classifier_img_1.keras')
