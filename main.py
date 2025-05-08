# --- Imports Section ---
import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt  
import seaborn as sns
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.applications.xception as xception
import zipfile
import sys
import time
import re
import cv2 # Added for Grad-CAM image processing

from PIL import Image
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D, GlobalAveragePooling2D, Lambda
from tensorflow.keras.layers import Normalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix # Added confusion_matrix

print('setup successful!')

# --- Constants ---
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 320
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3

base_path = "input/garbage-classification/garbage_classification/"
plot_dir = "plots" # Define plot_dir globally for use in multiple sections

# Dictionary to save our 12 classes
categories = {0: 'paper', 1: 'cardboard', 2: 'plastic', 3: 'metal', 4: 'trash', 5: 'battery',
              6: 'shoes', 7: 'clothes', 8: 'green-glass', 9: 'brown-glass', 10: 'white-glass',
              11: 'biological'}
class_names = list(categories.values()) # For confusion matrix and reports

print('defining constants successful!')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be set per GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        tf.config.list_physical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs\n")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
else:
    print("\nNo GPU detected by TensorFlow.")

# --- Function Definition ---
def add_class_name_prefix(df, col_name):
    df[col_name] = df[col_name].apply(lambda x: x[:re.search(r"\d", x).start()] + '/' + x)
    return df

# --- Grad-CAM Helper Functions ---
def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.utils.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.utils.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Load the original image
    img = keras.utils.load_img(img_path)
    img = keras.utils.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = plt.cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)
    # print(f"Grad-CAM saved to {cam_path}")
    return superimposed_img
# --- End Grad-CAM Helpers ---


# --- Data Loading and Preparation ---
if not os.path.isdir(plot_dir):
    os.makedirs(plot_dir)
    print(f"Created directory: {plot_dir}")

filenames_list = []
categories_list = []

if not os.path.isdir(base_path):
    print(f"Error: Base path '{base_path}' not found. Please check the path.\n")
    sys.exit(1)

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

df = pd.DataFrame({'filename': filenames_list, 'category': categories_list})
df = add_class_name_prefix(df, 'filename')
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print('number of elements = ' , len(df), "\n")
print(df.head(5), "\n")

# Visualization of Original Full Dataset Distribution
df_visualization = df.copy()
df_visualization['category_name'] = df_visualization['category'].replace(categories)
plt.figure(figsize=(12, 6))
df_visualization['category_name'].value_counts().plot.bar()
plt.xlabel("Garbage Classes", labelpad=14)
plt.ylabel("Images Count", labelpad=14)
plt.title("Count of images per class (Full Dataset)", y=1.02)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plot_filename = os.path.join(plot_dir, "full_dataset_class_distribution.png")
plt.savefig(plot_filename)
print(f"Plot saved to {plot_filename}")
plt.close()

# --- Data Splitting ---
df["category_name"] = df["category"].replace(categories)
train_df, validate_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['category_name'])
validate_df, test_df = train_test_split(validate_df, test_size=0.5, random_state=42, stratify=validate_df['category_name'])
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# --- Addressing Class Imbalance in train_df by Oversampling (Method 1) ---
print("\n--- Addressing Class Imbalance for Training Data ---")
print("Original training data distribution (before oversampling):")
print(train_df['category_name'].value_counts())
plt.figure(figsize=(12, 6))
train_df['category_name'].value_counts().plot.bar()
plt.title("Count of images per class (Training Set - Before Oversampling)", y=1.02)
plt.xticks(rotation=45, ha='right'); plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "train_class_dist_before_oversample.png"))
plt.close()

if not train_df.empty:
    majority_class_count = train_df['category_name'].value_counts().max()
    augmented_dfs = [train_df.copy()]
    for class_label, count in train_df['category_name'].value_counts().items():
        if count < majority_class_count:
            minority_class_df = train_df[train_df['category_name'] == class_label]
            samples_to_add = majority_class_count - count
            if not minority_class_df.empty and samples_to_add > 0:
                oversampled_rows = minority_class_df.sample(n=samples_to_add, replace=True, random_state=42)
                augmented_dfs.append(oversampled_rows)
    if len(augmented_dfs) > 1: train_df_balanced = pd.concat(augmented_dfs, ignore_index=True)
    else: train_df_balanced = train_df
    train_df = train_df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    print("\nBalanced training data distribution (after oversampling):")
    print(train_df['category_name'].value_counts())
    plt.figure(figsize=(12, 6))
    train_df['category_name'].value_counts().plot.bar()
    plt.title("Count of images per class (Training Set - After Oversampling)", y=1.02)
    plt.xticks(rotation=45, ha='right'); plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "train_class_dist_after_oversample.png"))
    plt.close()
else: print("Training dataframe is empty, skipping oversampling.")

total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
total_test = test_df.shape[0]
print(f'\nNew train size after oversampling = {total_train}, validate size = {total_validate}, test size = {total_test}\n')

# --- Model Definition ---
weights_path = 'input/xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'
if not os.path.exists(weights_path):
     print(f"Warning: Xception weights file not found at '{weights_path}'. Model will initialize without pre-trained weights.\n")
     weights_path = None
else: print(f"Found Xception weights.'\n")

xception_model_base = xception.Xception(include_top=False,
                                   input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS),
                                   weights=weights_path)
xception_model_base.trainable = False # Freeze weights

model = Sequential(name="Garbage_Classifier_Xception")
model.add(keras.Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(Lambda(xception.preprocess_input, name="xception_preprocessing")) # Use Xception's own preprocessor
model.add(xception_model_base) # Add the base model
model.add(GlobalAveragePooling2D(name="global_average_pooling"))
model.add(Dense(len(categories), activation='softmax', name="output_dense"))
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['categorical_accuracy']) # Use Keras Adam
model.summary()

# --- Callbacks ---
early_stop = EarlyStopping(patience=5, verbose=1, monitor='val_categorical_accuracy', mode='max', min_delta=0.001, restore_best_weights=True)
callbacks = [early_stop]
print('Callbacks defined!\n')

# --- Data Generators ---
batch_size = 32
train_datagen = image.ImageDataGenerator(
    rotation_range=40, shear_range=0.2, zoom_range=0.3, horizontal_flip=True,
    vertical_flip=True, width_shift_range=0.2, height_shift_range=0.2,
    brightness_range=[0.8,1.2], fill_mode='nearest'
)
train_generator = train_datagen.flow_from_dataframe(
    train_df, base_path, x_col='filename', y_col='category_name',
    target_size=IMAGE_SIZE, class_mode='categorical', batch_size=batch_size, shuffle=True
)
validation_datagen = image.ImageDataGenerator() # No augmentation for val/test
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, base_path, x_col='filename', y_col='category_name',
    target_size=IMAGE_SIZE, class_mode='categorical', batch_size=batch_size, shuffle=False
)
test_datagen = image.ImageDataGenerator()
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df, directory=base_path, x_col='filename', y_col='category_name',
    target_size=IMAGE_SIZE, color_mode="rgb", class_mode="categorical", batch_size=1, shuffle=False
)

# --- Model Training ---
EPOCHS = 20 # Keep low for quick testing; increase for actual training (e.g., 20-50)
num_workers = min(os.cpu_count() // 2 if os.cpu_count() else 1, 8) # Cap at 8, ensure at least 1
if num_workers == 0: num_workers = 1 # Ensure at least one worker
print(f"Using {num_workers} workers for data loading.")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    steps_per_epoch=max(1, total_train // batch_size),
    validation_steps=max(1, total_validate // batch_size),
    callbacks=callbacks,
    workers=num_workers,  # Number of parallel processes for data loading
    use_multiprocessing=True, # Enable multiprocessing
    max_queue_size=20  # Max size for the generator queue (e.g., 10-20 batches)
)

# --- Plotting History ---
if history and history.history:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(history.history['loss'], color='b', label="Training loss")
    ax1.plot(history.history['val_loss'], color='r', label="Validation loss")
    ax1.set_ylabel('Loss'); ax1.set_title('Training and Validation Loss'); ax1.legend()
    ax2.plot(history.history['categorical_accuracy'], color='b', label="Training accuracy")
    ax2.plot(history.history['val_categorical_accuracy'], color='r',label="Validation accuracy")
    ax2.set_ylabel('Accuracy'); ax2.set_title('Training and Validation Accuracy'); ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "training_history.png"))
    print(f"Training history plot saved to {os.path.join(plot_dir, 'training_history.png')}")
    plt.close(fig)
else: print("Training did not occur or history was not recorded.\n")


# --- Evaluation ---
print("\n--- Model Evaluation on Test Set ---")
if test_generator and test_df.shape[0] > 0:
    filenames_test = test_generator.filenames
    nb_samples_test = len(filenames_test)

    if nb_samples_test > 0:
        # 1. Overall Accuracy
        loss, accuracy = model.evaluate(test_generator, steps=nb_samples_test, verbose=1)
        print(f"\nOverall Test Accuracy: {accuracy * 100:.2f}%")
        print(f"Overall Test Loss: {loss:.4f}")

        # Get Predictions for other metrics
        # Ensure test_generator is reset if it has been used before (model.evaluate does this)
        # test_generator.reset() # Usually not needed if evaluate was just called
        preds_proba = model.predict(test_generator, steps=nb_samples_test, verbose=1)
        preds_indices = np.argmax(preds_proba, axis=1)

        # Map predictions to class names
        gen_label_map_inv = {v: k for k, v in test_generator.class_indices.items()}
        preds_names = [gen_label_map_inv[idx] for idx in preds_indices]
        
        true_labels_indices = test_generator.classes # Integer labels from generator
        true_labels_names = [gen_label_map_inv[idx] for idx in true_labels_indices]
        
        # Ensure correct true labels are used - test_df order is preserved due to shuffle=False
        # true_labels_from_df = test_df['category_name'].tolist()[:nb_samples_test]
        # if not all(t1 == t2 for t1, t2 in zip(true_labels_names, true_labels_from_df)):
        #    print("Warning: Mismatch between generator true labels and dataframe true labels. Using generator labels.")

        # 2. Precision, Recall, F1-Score (Per Class)
        print("\nClassification Report (Test Set):\n")
        # Use class_names (sorted list of your category names) for consistent report ordering
        report = classification_report(true_labels_names, preds_names, labels=class_names, zero_division=0, target_names=class_names)
        print(report)
        with open(os.path.join(plot_dir, "classification_report.txt"), "w") as f:
            f.write(report)
        print(f"Classification report saved to {os.path.join(plot_dir, 'classification_report.txt')}")


        # 3. Confusion Matrix
        print("\nGenerating Confusion Matrix...")
        cm = confusion_matrix(true_labels_names, preds_names, labels=class_names)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.title("Confusion Matrix (Test Set)")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        cm_path = os.path.join(plot_dir, "confusion_matrix.png")
        plt.savefig(cm_path)
        print(f"Confusion matrix saved to {cm_path}")
        plt.close()

        # 4. Grad-CAM Heatmaps
        print("\nGenerating Grad-CAM Heatmaps for a few test images...")
        # Find the last convolutional layer in the Xception base model
        # Typically 'block14_sepconv2_act' for Xception, but verify with model.summary() if needed.
        # We need to access it through the Sequential model: model.get_layer(xception_model_base.name).get_layer(last_conv_layer_name)
        last_conv_layer_name_in_base = "block14_sepconv2_act" # Common for Xception
        
        # We need the preprocessed image array for Grad-CAM
        # Xception's preprocess_input is already part of the model as a Lambda layer.

        num_gradcam_examples = 5
        gradcam_dir = os.path.join(plot_dir, "grad_cam_examples")
        if not os.path.exists(gradcam_dir):
            os.makedirs(gradcam_dir)

        # Get some correctly and incorrectly classified examples
        correctly_classified_indices = [i for i, (t, p) in enumerate(zip(true_labels_names, preds_names)) if t == p]
        incorrectly_classified_indices = [i for i, (t, p) in enumerate(zip(true_labels_names, preds_names)) if t != p]
        
        selected_indices = []
        if correctly_classified_indices:
            selected_indices.extend(random.sample(correctly_classified_indices, min(num_gradcam_examples // 2, len(correctly_classified_indices))))
        if incorrectly_classified_indices:
             selected_indices.extend(random.sample(incorrectly_classified_indices, min(num_gradcam_examples - len(selected_indices), len(incorrectly_classified_indices))))
        
        # If still not enough, pick randomly
        if len(selected_indices) < num_gradcam_examples and len(true_labels_names) >= num_gradcam_examples :
            remaining_needed = num_gradcam_examples - len(selected_indices)
            potential_indices = [i for i in range(len(true_labels_names)) if i not in selected_indices]
            if potential_indices:
                 selected_indices.extend(random.sample(potential_indices, min(remaining_needed, len(potential_indices))))


        for i in selected_indices:
            img_filename = test_df['filename'].iloc[i] # Get filename from test_df
            img_path = os.path.join(base_path, img_filename)
            
            # Original image needs to be preprocessed in the same way as training
            # For GradCAM, we need the array that goes INTO the model
            img_array_for_gradcam = get_img_array(img_path, size=IMAGE_SIZE) # This is (1, H, W, C)
            
            # The xception.preprocess_input is ALREADY IN THE MODEL, so we don't apply it manually here to img_array_for_gradcam
            # The `make_gradcam_heatmap` function will use the full model which includes preprocessing.

            try:
                # Access the last conv layer within the xception_model_base which is inside our Sequential model
                full_last_conv_layer_name = f"{xception_model_base.name}/{last_conv_layer_name_in_base}"
                
                heatmap = make_gradcam_heatmap(img_array_for_gradcam, model, full_last_conv_layer_name, pred_index=preds_indices[i])
                
                superimposed_img_pil = save_and_display_gradcam(img_path, heatmap, alpha=0.5)
                
                # Plot original, heatmap, and superimposed
                fig_gc, axes_gc = plt.subplots(1, 3, figsize=(15, 5))
                original_pil_img = Image.open(img_path)
                axes_gc[0].imshow(original_pil_img)
                axes_gc[0].set_title(f"Original: {true_labels_names[i]}\nPred: {preds_names[i]}")
                axes_gc[0].axis('off')

                axes_gc[1].imshow(heatmap)
                axes_gc[1].set_title("Grad-CAM Heatmap")
                axes_gc[1].axis('off')

                axes_gc[2].imshow(superimposed_img_pil)
                axes_gc[2].set_title("Superimposed")
                axes_gc[2].axis('off')
                
                plt.tight_layout()
                grad_cam_filename = f"gradcam_{i}_{os.path.basename(img_filename).replace('.jpg','')}.png"
                plt.savefig(os.path.join(gradcam_dir, grad_cam_filename))
                plt.close(fig_gc)
                print(f"Saved Grad-CAM for {img_filename} (True: {true_labels_names[i]}, Pred: {preds_names[i]})")

            except Exception as e:
                print(f"Error generating Grad-CAM for {img_filename}: {e}")
                # This might happen if the layer name is incorrect or other issues.
                # Check `model.summary()` and `xception_model_base.summary()` for correct layer names.

        # 5. Inference Time (Optional)
        print("\nMeasuring Inference Time...")
        num_inference_samples = min(50, nb_samples_test) # Use up to 50 samples or all test samples if fewer
        inference_times = []
        
        # Create a small temporary generator or get image arrays
        # To avoid re-creating generator, let's get some image paths and preprocess them
        inference_image_paths = [os.path.join(base_path, test_df['filename'].iloc[i]) for i in range(num_inference_samples)]

        for i in range(num_inference_samples):
            # img_array_inf = test_generator[i][0] # This also works if generator is reset
            img_array_inf = get_img_array(inference_image_paths[i], IMAGE_SIZE)
            # Preprocessing is part of the model, so feed raw pixels (scaled 0-255) to model.predict
            
            start_time = time.time()
            _ = model.predict(img_array_inf, verbose=0) # verbose=0 to suppress progress bar
            end_time = time.time()
            inference_times.append(end_time - start_time)
        
        if inference_times:
            avg_inference_time = np.mean(inference_times)
            print(f"Average inference time per image ({num_inference_samples} samples): {avg_inference_time * 1000:.2f} ms")
            print(f"FPS: {1/avg_inference_time:.2f}")
        else:
            print("Could not measure inference time (no samples).")

    else:
        print("Test generator is empty (nb_samples_test = 0). Skipping evaluation metrics.\n")
else:
    print("Test generator could not be created or test_df is empty. Skipping evaluation and prediction.\n")

print("\nScript finished.")