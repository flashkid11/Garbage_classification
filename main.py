# --- Imports Section ---
import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns
# Use tensorflow.keras consistently
import tensorflow as tf
import tensorflow.keras as keras # Keep this alias if used extensively
import tensorflow.keras.applications.xception as xception
import zipfile
import sys
import time
import re

from PIL import Image
# Use tensorflow.keras paths
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D, GlobalAveragePooling2D, Lambda
from tensorflow.keras.layers import Normalization # Corrected import path
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

print('setup successful!')

# --- Constants ---
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 320
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3

base_path = "input/garbage-classification/garbage_classification/"

# Dictionary to save our 12 classes
categories = {0: 'paper', 1: 'cardboard', 2: 'plastic', 3: 'metal', 4: 'trash', 5: 'battery',
              6: 'shoes', 7: 'clothes', 8: 'green-glass', 9: 'brown-glass', 10: 'white-glass',
              11: 'biological'}

print('defining constants successful!')

# --- Function Definition ---
# Add class name prefix to filename. So for example "/paper104.jpg" become "paper/paper104.jpg"
def add_class_name_prefix(df, col_name):
    # Use raw string for regex pattern
    df[col_name] = df[col_name].apply(lambda x: x[:re.search(r"\d", x).start()] + '/' + x)
    return df

# --- Data Loading and Preparation ---
filenames_list = []
categories_list = []

# Check if base_path exists before proceeding
if not os.path.isdir(base_path):
    print(f"Error: Base path '{base_path}' not found. Please check the path.\n")
    sys.exit(1) # Exit if the base data directory doesn't exist

for category_num, category_name in categories.items():
    folder_path = os.path.join(base_path, category_name)
    if not os.path.isdir(folder_path):
        print(f"Warning: Directory not found for category '{category_name}' at path '{folder_path}'. Skipping.\n")
        continue
    filenames = os.listdir(folder_path)
    filenames_list.extend(filenames)
    categories_list.extend([category_num] * len(filenames))

if not filenames_list:
    print(f"Error: No image files found in the subdirectories of '{base_path}'. Please check the data structure.\n")
    sys.exit(1)

df = pd.DataFrame({
    'filename': filenames_list,
    'category': categories_list
})

df = add_class_name_prefix(df, 'filename')

# Shuffle the dataframe
df = df.sample(frac=1).reset_index(drop=True)

print('number of elements = ' , len(df), "\n")

print(df.head(5), "\n")

# --- Visualization ---
df_visualization = df.copy()
df_visualization['category'] = df_visualization['category'].apply(lambda x:categories[x])
plt.figure(figsize=(12, 6)) # Adjust figure size for better label visibility
df_visualization['category'].value_counts().plot.bar(x = 'category', y = 'count') # Swapped x and y for clarity
plt.xlabel("Garbage Classes", labelpad=14)
plt.ylabel("Images Count", labelpad=14)
plt.title("Count of images per class", y=1.02)
plt.xticks(rotation=45, ha='right') # Rotate labels for readability
plt.tight_layout() # Adjust layout
plt.show() # Display the plot - uncomment if running interactively

# --- Model Definition ---
# Ensure the weights path is correct for your environment
weights_path = '../input/xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'
if not os.path.exists(weights_path):
     print(f"Warning: Xception weights file not found at '{weights_path}'. Model will initialize without pre-trained weights.\n")
     weights_path = None # Set to None if file doesn't exist, Xception will use random weights or download if 'imagenet' is specified
else:
     print(f"Found Xception weights at '{weights_path}'\n")


xception_layer = xception.Xception(include_top=False,
                                   input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS),
                                   weights=weights_path) # Use variable path, handles case where file doesn't exist

xception_layer.trainable = False # Freeze weights

model = Sequential()
model.add(keras.Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))

# Create a custom layer to apply the preprocessing
def xception_preprocessing(img):
  return xception.preprocess_input(img)

model.add(Lambda(xception_preprocessing))

model.add(xception_layer)
model.add(GlobalAveragePooling2D()) # Use TF/Keras directly
model.add(Dense(len(categories), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

model.summary()

# --- Callbacks ---
early_stop = EarlyStopping(patience=2, verbose=1, monitor='val_categorical_accuracy', mode='max', min_delta=0.001, restore_best_weights=True)
callbacks = [early_stop]
print('Callbacks defined!\n')

# --- Data Splitting ---
# Change the categories from numbers to names for the generator
df["category"] = df["category"].replace(categories)

train_df, validate_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['category']) # Added stratify
validate_df, test_df = train_test_split(validate_df, test_size=0.5, random_state=42, stratify=validate_df['category']) # Added stratify

train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
total_test = test_df.shape[0]

print(f'train size = {total_train}, validate size = {total_validate}, test size = {total_test}\n')

# --- Data Generators ---
batch_size = 32 # Reduced batch size slightly, maybe helpful if memory is constrained

# Consider adding rescaling if Xception preprocess_input doesn't handle it (it usually does)
# rescale=1./255 # Add this if needed
train_datagen = image.ImageDataGenerator(
    # Augmentation is commented out, uncomment if needed
    # rotation_range=30,
    # shear_range=0.1,
    # zoom_range=0.3,
    # horizontal_flip=True,
    # vertical_flip = True,
    # width_shift_range=0.2,
    # height_shift_range=0.2
    # rescale=rescale # Add if needed
)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    base_path,
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)

# Use the same ImageDataGenerator for validation, typically without augmentation
validation_datagen = image.ImageDataGenerator() # rescale=rescale if needed

validation_generator = validation_datagen.flow_from_dataframe(
    validate_df,
    base_path,
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=False # No need to shuffle validation data
)

# --- Model Training ---
EPOCHS = 2 # Increased epochs slightly, early stopping will prevent overfitting
history = model.fit( # Use model.fit instead of fit_generator
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    # steps_per_epoch and validation_steps are often inferred but can be specified
    steps_per_epoch=total_train // batch_size,
    validation_steps=total_validate // batch_size,
    callbacks=callbacks
)

# --- Plotting History ---
if history and history.history: # Check if training happened and history exists
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8)) # Make plot larger
    ax1.plot(history.history['loss'], color='b', label="Training loss")
    ax1.plot(history.history['val_loss'], color='r', label="Validation loss")
    # ax1.set_yticks(np.arange(0, 0.7, 0.1)) # Maybe adjust range based on actual loss values
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()

    ax2.plot(history.history['categorical_accuracy'], color='b', label="Training accuracy")
    ax2.plot(history.history['val_categorical_accuracy'], color='r',label="Validation accuracy")
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.show()
else:
    print("Training did not occur or history was not recorded.\n")


# --- Evaluation ---
test_datagen = image.ImageDataGenerator() # rescale=rescale if needed

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=base_path,
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    color_mode="rgb",
    class_mode="categorical",
    batch_size=1, # Evaluate one by one for simplicity here, but larger batch is faster
    shuffle=False
)

filenames = test_generator.filenames
nb_samples = len(filenames)

if nb_samples > 0:
    # Use model.evaluate instead of evaluate_generator
    loss, accuracy = model.evaluate(test_generator, steps=nb_samples) # Ensure steps covers all samples
    print(f'Accuracy on test set = {accuracy * 100:.2f}%\n')

    # --- Predictions and Classification Report ---
    # Need generator mapping for report
    gen_label_map = test_generator.class_indices
    gen_label_map_inv = {v: k for k, v in gen_label_map.items()} # Invert map {index: name}
    print("Generator Label Map:", gen_label_map_inv, "\n")

    # Use model.predict instead of predict_generator
    preds_proba = model.predict(test_generator, steps=nb_samples)
    preds_indices = np.argmax(preds_proba, axis=1) # Get index of max probability

    # Map predicted indices to class names
    preds_names = [gen_label_map_inv[idx] for idx in preds_indices]

    # Get true labels (already strings from the dataframe)
    true_labels = test_df['category'].tolist() # Ensure it's a list of strings

    # Ensure lengths match
    if len(true_labels) == len(preds_names):
        print("\nClassification Report:\n")
        print(classification_report(true_labels, preds_names), "\n")
    else:
        print(f"Error: Mismatch in number of true labels ({len(true_labels)}) and predictions ({len(preds_names)}).\n")

else:
    print("Test generator is empty. Skipping evaluation and prediction.\n")