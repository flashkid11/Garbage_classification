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
import cv2
import traceback

from PIL import Image
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D, GlobalAveragePooling2D, Lambda
from tensorflow.keras.layers import Normalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++  TEST MODE FLAG +++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
TEST_MODE = True # Set to False for actual training
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

print('setup successful!')

# --- Constants ---
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 320
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3

base_path = "input/garbage-classification/garbage_classification/" # Adjust if your data is elsewhere
plot_dir = "plots"

categories = {0: 'paper', 1: 'cardboard', 2: 'plastic', 3: 'metal', 4: 'trash', 5: 'battery',
              6: 'shoes', 7: 'clothes', 8: 'green-glass', 9: 'brown-glass', 10: 'white-glass',
              11: 'biological'}
class_names = list(categories.values())
CLASS_NAMES_FULL = class_names # Use this consistently for all generators

print('defining constants successful!')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs\n")
    except RuntimeError as e:
        print(e)
else:
    print("\nNo GPU detected by TensorFlow.")

def add_class_name_prefix(df, col_name):
    def prefix_logic(x):
        match = re.search(r"\d", x)
        # Ensure the filename itself doesn't already contain a slash (indicating it might be pre-formatted)
        if '/' in x:
            return x
        if match:
            # Assumes the part before the first digit is the category folder name
            return x[:match.start()] + '/' + x
        else:
            # If no digit, attempt to match against known category names if it doesn't already look like a path
            # This is a fallback and might need refinement based on actual filename patterns
            for cat_name_key, cat_name_val in categories.items():
                if x.lower().startswith(cat_name_val.lower()):
                    return cat_name_val + '/' + x
            print(f"Warning: Could not determine prefix for filename '{x}'. Returning as is.")
            return x
    df[col_name] = df[col_name].apply(prefix_logic)
    return df

def get_img_array(img_path, size):
    img = keras.utils.load_img(img_path, target_size=size)
    array = keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    img = keras.utils.load_img(img_path)
    img = keras.utils.img_to_array(img)
    heatmap = np.uint8(255 * heatmap)
    jet = plt.cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)
    superimposed_img.save(cam_path)
    return superimposed_img

if not os.path.isdir(plot_dir):
    os.makedirs(plot_dir)
    print(f"Created directory: {plot_dir}")

filenames_list = []
categories_list = [] # Will store integer category indices
if not os.path.isdir(base_path):
    print(f"Error: Base path '{base_path}' not found. Please check the path.\n")
    if TEST_MODE:
        print("TEST_MODE: Creating dummy base_path and a few image files for testing flow.")
        os.makedirs(base_path, exist_ok=True)
        # Create dummy files for a subset of categories to ensure TEST_MODE can run
        dummy_categories_to_create = CLASS_NAMES_FULL[:max(4, len(CLASS_NAMES_FULL)//3)] # Create at least 4 or 1/3rd
        print(f"Creating dummy files for categories: {dummy_categories_to_create}")
        for cat_name in dummy_categories_to_create:
            cat_folder = os.path.join(base_path, cat_name)
            os.makedirs(cat_folder, exist_ok=True)
            for i in range(5): # Create 5 dummy files per category
                try:
                    dummy_img = Image.new('RGB', (60, 30), color = 'red')
                    # Filename should be simple, prefixing happens later or is handled by flow_from_dataframe x_col
                    dummy_img_name = f"{cat_name.replace('-', '_')}_{i+1}.jpg" # Make filename a bit more unique
                    dummy_img.save(os.path.join(cat_folder, dummy_img_name))
                except Exception as e_dummy:
                    print(f"Error creating dummy file in {cat_folder}: {e_dummy}")
        if not os.path.isdir(base_path): sys.exit(1)
    else:
        sys.exit(1)

for category_idx, category_name_val in categories.items(): # Use .items() to get index and name
    folder_path = os.path.join(base_path, category_name_val)
    if not os.path.isdir(folder_path):
        # print(f"Warning: Folder for category '{category_name_val}' not found at '{folder_path}'. Skipping.")
        continue
    filenames_in_folder = os.listdir(folder_path)
    # For flow_from_dataframe, 'filename' column should be 'category_folder/actual_filename.jpg'
    filenames_list.extend([os.path.join(category_name_val, fn) for fn in filenames_in_folder])
    categories_list.extend([category_idx] * len(filenames_in_folder)) # Store numeric category

if not filenames_list:
    print(f"Error: No image files found in '{base_path}' or its subdirectories. Please check data structure.\n")
    sys.exit(1)

df = pd.DataFrame({'filename': filenames_list, 'category_num': categories_list})
# 'category_name' column is derived from 'category_num' for stratification and y_col in generator
df['category_name'] = df['category_num'].map(categories)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print('Total elements loaded = ' , len(df), "\n")
# print(df.head()) # Useful for debugging

df_visualization = df.copy()
plt.figure(figsize=(12, 6)); df_visualization['category_name'].value_counts().plot.bar()
plt.title("Count of images per class (Full Dataset)"); plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "full_dataset_class_distribution.png")); plt.close()

# Stratify by 'category_name' as it's string and often used for that
min_total_samples_for_split = 2 * len(CLASS_NAMES_FULL)
if len(df) < min_total_samples_for_split and len(df) > 0:
    print(f"Warning: Dataset too small ({len(df)} samples) for reliable stratified train/validate/test split.")
    # Simplified split for tiny datasets
    if len(df) > 3:
        train_df, temp_df = train_test_split(df, test_size=min(0.4, (2/len(df)) if len(df) > 0 else 0.4), random_state=42, stratify=df['category_name'] if len(df['category_name'].unique()) > 1 else None)
        if len(temp_df) > 1:
             validate_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['category_name'] if len(temp_df['category_name'].unique()) > 1 else None)
        else:
            validate_df = temp_df.copy() if not temp_df.empty else pd.DataFrame(columns=df.columns)
            test_df = temp_df.copy() if not temp_df.empty else pd.DataFrame(columns=df.columns)
    else:
        train_df = df.copy()
        validate_df = df.copy() if len(df)>1 else pd.DataFrame(columns=df.columns)
        test_df = df.copy() if len(df)>1 else pd.DataFrame(columns=df.columns)
else: # Original split logic if enough data
    train_df, validate_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['category_name'])
    validate_df, test_df = train_test_split(validate_df, test_size=0.5, random_state=42, stratify=validate_df['category_name'])

train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


# ++++++++++++++ TEST MODE: SUBSET DATA ++++++++++++++
if TEST_MODE:
    print("\n!!! TEST MODE ENABLED: Using subset of data and minimal training !!!\n")
    test_batch_size = 2
    min_samples_per_split_total = test_batch_size * 2 # e.g., 4 for train

    # Function to safely sample/subset a dataframe
    def safe_subset(input_df, target_size, batch_size_for_strat):
        if input_df.empty: return input_df
        num_unique_classes = len(input_df['category_name'].unique())
        if num_unique_classes == 0: return input_df # No classes to stratify by

        # Ensure at least 1 sample per class if possible, up to target_size
        samples_per_class = max(1, target_size // num_unique_classes if num_unique_classes > 0 else 1)
        
        # Stratified sampling: try to get samples_per_class from each category
        output_df = input_df.groupby('category_name', group_keys=False).apply(lambda x: x.sample(min(len(x), samples_per_class), random_state=42, replace=len(x) < samples_per_class) )
        output_df = output_df.sample(frac=1, random_state=42).reset_index(drop=True) # Shuffle
        
        # If still below target_size (e.g. few unique classes), oversample to meet target_size or batch_size
        if len(output_df) < target_size and not output_df.empty:
            output_df = output_df.sample(n=target_size, replace=True, random_state=42).reset_index(drop=True)
        elif len(output_df) > target_size:
             output_df = output_df.iloc[:target_size] # Cap total size

        # Ensure total samples is at least batch_size if df is not empty
        if not output_df.empty and len(output_df) < batch_size_for_strat:
             output_df = output_df.sample(n=batch_size_for_strat, replace=True, random_state=42).reset_index(drop=True)
        return output_df

    train_df = safe_subset(train_df, min_samples_per_split_total, test_batch_size)
    validate_df = safe_subset(validate_df, test_batch_size, test_batch_size) # Target one batch for validation
    test_df = safe_subset(test_df, max(test_batch_size, min(10, len(df)//3 if len(df)>0 else 1)), 1) # Test batch size is 1

    print(f"TEST MODE Active: Sliced train_df to {len(train_df)} samples.")
    print(f"TEST MODE Active: Sliced validate_df to {len(validate_df)} samples.")
    print(f"TEST MODE Active: Sliced test_df to {len(test_df)} samples.")
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++

print("\n--- Addressing Class Imbalance for Training Data ---")
print("Training data distribution (before oversampling):")
if not train_df.empty: print(train_df['category_name'].value_counts())
else: print("Train DF is empty.")

if not train_df.empty:
    if not TEST_MODE: # Only do extensive oversampling if not in TEST_MODE
        majority_class_count = train_df['category_name'].value_counts().max()
        if majority_class_count > 0 :
            augmented_dfs = [train_df.copy()]
            for class_label, count in train_df['category_name'].value_counts().items():
                if count < majority_class_count and count > 0 :
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
    elif TEST_MODE:
        print("TEST MODE: Skipping full oversampling logic, using potentially imbalanced small train_df.")
else: print("Training dataframe is empty, skipping oversampling.")

total_train = len(train_df)
total_validate = len(validate_df)
total_test = len(test_df)
print(f'\nEffective train size = {total_train}, validate size = {total_validate}, test size = {total_test}\n')

weights_path = 'input/xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'
if not os.path.exists(weights_path):
     print(f"Warning: Xception weights file not found at '{weights_path}'. Model will use random weights.\n")
     weights_path = None
else: print(f"Found Xception weights at '{weights_path}'.\n")

xception_model_base = xception.Xception(
    include_top=False,
    input_shape=IMAGE_SIZE+(IMAGE_CHANNELS,),
    weights=weights_path
)
xception_model_base_name = xception_model_base.name # Get actual name, e.g., "xception"
print(f"Xception base model instantiated with name: '{xception_model_base_name}'")
xception_model_base.trainable = False

model = Sequential(name="Garbage_Classifier_Xception")
model.add(keras.Input(shape=IMAGE_SIZE+(IMAGE_CHANNELS,), name="input_image"))
model.add(Lambda(xception.preprocess_input, name="xception_preprocessing"))
model.add(xception_model_base)
model.add(GlobalAveragePooling2D(name="global_average_pooling"))
model.add(Dense(len(CLASS_NAMES_FULL), activation='softmax', name="output_dense")) # Output units = num total classes
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['categorical_accuracy'])
model.summary(line_length=120)

early_stop = EarlyStopping(patience=5, verbose=1, monitor='val_categorical_accuracy', mode='max', min_delta=0.001, restore_best_weights=True)
callbacks = [early_stop]

batch_size_to_use = test_batch_size if TEST_MODE and 'test_batch_size' in locals() else 32

# --- Data Generators ---
train_generator, validation_generator, test_generator = None, None, None

# Train Generator
if not train_df.empty:
    unknown_classes = set(train_df['category_name'].unique()) - set(CLASS_NAMES_FULL)
    if unknown_classes:
        print(f"ERROR: train_df contains unknown classes: {unknown_classes}. Exiting.")
        sys.exit(1)
    print(f"Train generator using classes: {CLASS_NAMES_FULL}")
    train_datagen = image.ImageDataGenerator(
        rotation_range=40, shear_range=0.2, zoom_range=0.3, horizontal_flip=True,
        vertical_flip=True, width_shift_range=0.2, height_shift_range=0.2,
        brightness_range=[0.8,1.2], fill_mode='nearest'
    )
    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        base_path, # Directory where subfolders (named by class) for images are located
        x_col='filename', # e.g. "paper/paper1.jpg" - path relative to base_path
        y_col='category_name',
        target_size=IMAGE_SIZE,
        class_mode='categorical',
        batch_size=batch_size_to_use,
        shuffle=True,
        classes=CLASS_NAMES_FULL # CRITICAL: Use full list for consistent encoding
    )
else:
    print("ERROR: train_df is empty. Cannot create train_generator.")
    # sys.exit(1) # Decide if to exit or allow script to continue if other parts can run

# Validation Generator
if not validate_df.empty:
    unknown_classes_val = set(validate_df['category_name'].unique()) - set(CLASS_NAMES_FULL)
    if unknown_classes_val:
        print(f"ERROR: validate_df contains unknown classes: {unknown_classes_val}. Exiting.")
        sys.exit(1)
    print(f"Validation generator using classes: {CLASS_NAMES_FULL}")
    validation_datagen = image.ImageDataGenerator() # No augmentation for validation
    validation_generator = validation_datagen.flow_from_dataframe(
        validate_df,
        base_path,
        x_col='filename',
        y_col='category_name',
        target_size=IMAGE_SIZE,
        class_mode='categorical',
        batch_size=batch_size_to_use,
        shuffle=False,
        classes=CLASS_NAMES_FULL
    )
else:
    print("WARNING: validate_df is empty. Validation will be skipped if generator is None.")

# Test Generator
if not test_df.empty:
    unknown_classes_test = set(test_df['category_name'].unique()) - set(CLASS_NAMES_FULL)
    if unknown_classes_test:
        print(f"ERROR: test_df contains unknown classes: {unknown_classes_test}. Exiting or check data.")
        # sys.exit(1)
    print(f"Test generator using classes: {CLASS_NAMES_FULL}")
    test_datagen = image.ImageDataGenerator()
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=base_path,
        x_col='filename',
        y_col='category_name',
        target_size=IMAGE_SIZE,
        color_mode="rgb",
        class_mode="categorical",
        batch_size=1, # Usually 1 for evaluation and Grad-CAM
        shuffle=False,
        classes=CLASS_NAMES_FULL
    )
else:
    print("WARNING: test_df is empty. Test evaluation will be skipped.")


# --- Model Training ---
EPOCHS_FULL = 20
num_workers = min(os.cpu_count() // 2 if os.cpu_count() else 1, 8)
if num_workers == 0: num_workers = 1

if TEST_MODE:
    print(f"TEST MODE: Training for 1 epoch.")
    EPOCHS_TO_RUN = 1
    STEPS_PER_EPOCH_TEST = max(1, total_train // batch_size_to_use if total_train > 0 and batch_size_to_use > 0 else 1)
    VALIDATION_STEPS_TEST = max(1, total_validate // batch_size_to_use if total_validate > 0 and batch_size_to_use > 0 else 1)
    actual_callbacks = None
    print(f"Using Steps per epoch: {STEPS_PER_EPOCH_TEST}, Validation steps: {VALIDATION_STEPS_TEST}")
else:
    EPOCHS_TO_RUN = EPOCHS_FULL
    STEPS_PER_EPOCH_TEST = max(1, total_train // batch_size_to_use if total_train > 0 and batch_size_to_use > 0 else 1)
    VALIDATION_STEPS_TEST = max(1, total_validate // batch_size_to_use if total_validate > 0 and batch_size_to_use > 0 else 1)
    actual_callbacks = callbacks

history = None
if train_generator is None or validation_generator is None :
    print("WARNING: Training or validation generator is None. Skipping model.fit().")
elif STEPS_PER_EPOCH_TEST == 0 or VALIDATION_STEPS_TEST == 0:
     print(f"WARNING: Steps per epoch ({STEPS_PER_EPOCH_TEST}) or validation steps ({VALIDATION_STEPS_TEST}) is zero. Skipping model.fit(). Check data and batch sizes.")
else:
    print(f"Starting model.fit with {EPOCHS_TO_RUN} epochs...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS_TO_RUN,
        validation_data=validation_generator,
        steps_per_epoch=STEPS_PER_EPOCH_TEST,
        validation_steps=VALIDATION_STEPS_TEST,
        callbacks=actual_callbacks,
        workers=num_workers,
        use_multiprocessing=True if num_workers > 1 else False,
        max_queue_size=20
    )

# --- Plotting History ---
if history and history.history:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(history.history['loss'], color='b', label="Training loss")
    if 'val_loss' in history.history: ax1.plot(history.history['val_loss'], color='r', label="Validation loss")
    ax1.set_ylabel('Loss'); ax1.set_title('Training and Validation Loss'); ax1.legend()
    ax2.plot(history.history['categorical_accuracy'], color='b', label="Training accuracy")
    if 'val_categorical_accuracy' in history.history: ax2.plot(history.history['val_categorical_accuracy'], color='r',label="Validation accuracy")
    ax2.set_ylabel('Accuracy'); ax2.set_title('Training and Validation Accuracy'); ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "training_history.png"))
    print(f"Training history plot saved to {os.path.join(plot_dir, 'training_history.png')}")
    plt.close(fig)
else: print("Training history not available.\n")


# --- Evaluation ---
print("\n--- Model Evaluation on Test Set ---")
if test_generator is None or total_test == 0:
    print("Test generator is None or test set is empty. Skipping evaluation.")
else:
    filenames_test = test_generator.filenames
    nb_samples_test = len(filenames_test) if filenames_test else 0

    if nb_samples_test > 0:
        print(f"Evaluating on {nb_samples_test} test samples...")
        eval_steps = max(1, nb_samples_test // test_generator.batch_size if test_generator.batch_size > 0 else nb_samples_test)

        loss, accuracy = model.evaluate(test_generator, steps=eval_steps, verbose=1)
        print(f"\nOverall Test Accuracy: {accuracy * 100:.2f}%")
        print(f"Overall Test Loss: {loss:.4f}")

        preds_proba = model.predict(test_generator, steps=eval_steps, verbose=1)
        preds_indices = np.argmax(preds_proba, axis=1)

        gen_label_map_inv = {v: k for k, v in test_generator.class_indices.items()}
        preds_names = [gen_label_map_inv[idx] for idx in preds_indices]

        true_labels_indices = test_generator.classes[:len(preds_indices)]
        true_labels_names = [gen_label_map_inv[idx] for idx in true_labels_indices]

        report_labels = list(test_generator.class_indices.keys()) # Classes the generator knows
        cm_labels = report_labels

        print("\nClassification Report (Test Set):\n")
        report = classification_report(true_labels_names, preds_names, labels=report_labels, zero_division=0, target_names=report_labels)
        print(report)

        print("\nGenerating Confusion Matrix...")
        cm = confusion_matrix(true_labels_names, preds_names, labels=cm_labels)
        plt.figure(figsize=(max(8, len(cm_labels)), max(6, len(cm_labels)*0.8)))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=cm_labels, yticklabels=cm_labels)
        plt.title("Confusion Matrix (Test Set)"); plt.ylabel("True Label"); plt.xlabel("Predicted Label")
        plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0); plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "confusion_matrix.png")); plt.close()

        # 4. Grad-CAM Heatmaps
        print("\nGenerating Grad-CAM Heatmaps for a few test images...")
        last_conv_layer_name_in_xception = "block12_sepconv1_act"

        num_gradcam_examples = min(3, nb_samples_test, len(preds_indices)) # Ensure not to exceed available predictions
        gradcam_dir = os.path.join(plot_dir, "grad_cam_examples")
        if not os.path.exists(gradcam_dir): os.makedirs(gradcam_dir)

        selected_indices = []
        if num_gradcam_examples > 0:
            try:
                selected_indices = random.sample(range(len(preds_indices)), num_gradcam_examples)
            except ValueError: # If len(preds_indices) is 0
                 print("Not enough predictions to sample for Grad-CAM.")
        else:
            print("No test samples or no Grad-CAM examples requested/possible.")

        grad_model_for_cam_builder = None
        if selected_indices:
            try:
                lambda_layer = model.get_layer(name="xception_preprocessing")
                # Use the actual name of the Xception model instance obtained earlier
                xception_sub_model_layer = model.get_layer(name=xception_model_base_name)

                input_to_xception_in_main_flow = lambda_layer.output
                internal_conv_submodel = Model(
                    inputs=xception_sub_model_layer.inputs,
                    outputs=xception_sub_model_layer.get_layer(last_conv_layer_name_in_xception).output
                )
                target_conv_output_in_main_flow = internal_conv_submodel(input_to_xception_in_main_flow)
                print(f"Successfully obtained target conv tensor for Grad-CAM from '{last_conv_layer_name_in_xception}'.")
                grad_model_for_cam_builder = Model(
                    model.inputs,
                    [target_conv_output_in_main_flow, model.output]
                )
                print("Grad-CAM model built successfully.")
            except Exception as e_grad_setup:
                print(f"Error during Grad-CAM setup: {e_grad_setup}")
                traceback.print_exc()
                grad_model_for_cam_builder = None

        if grad_model_for_cam_builder:
            for k_idx in selected_indices:
                img_rel_path = test_generator.filenames[k_idx]
                img_path = os.path.join(base_path, img_rel_path)
                if not os.path.exists(img_path):
                    print(f"Warning: Image path for Grad-CAM not found: {img_path}. Skipping.")
                    continue
                img_array_for_gradcam_raw = get_img_array(img_path, size=IMAGE_SIZE)
                try:
                    with tf.GradientTape() as tape:
                        conv_layer_output_value, preds_value_tape = grad_model_for_cam_builder(img_array_for_gradcam_raw)
                        pred_index_for_tape = preds_indices[k_idx]
                        class_channel = preds_value_tape[:, pred_index_for_tape]
                    grads_value = tape.gradient(class_channel, conv_layer_output_value)
                    if grads_value is None:
                        print(f"Warning: Gradients are None for image {img_rel_path}. Skipping Grad-CAM.")
                        continue
                    pooled_grads_value = tf.reduce_mean(grads_value, axis=(0, 1, 2))
                    conv_layer_output_value = conv_layer_output_value[0]
                    heatmap = conv_layer_output_value @ pooled_grads_value[..., tf.newaxis]
                    heatmap = tf.squeeze(heatmap)
                    heatmap_max = tf.math.reduce_max(heatmap)
                    if heatmap_max == 0: heatmap = tf.zeros_like(heatmap)
                    else: heatmap = tf.maximum(heatmap, 0) / heatmap_max
                    heatmap_numpy = heatmap.numpy()
                    superimposed_img_pil = save_and_display_gradcam(img_path, heatmap_numpy, alpha=0.5)
                    fig_gc, axes_gc = plt.subplots(1, 3, figsize=(15, 5))
                    original_pil_img = Image.open(img_path)
                    axes_gc[0].imshow(original_pil_img); axes_gc[0].set_title(f"Original: {true_labels_names[k_idx]}\nPred: {preds_names[k_idx]}"); axes_gc[0].axis('off')
                    axes_gc[1].imshow(heatmap_numpy); axes_gc[1].set_title("Grad-CAM Heatmap"); axes_gc[1].axis('off')
                    axes_gc[2].imshow(superimposed_img_pil); axes_gc[2].set_title("Superimposed"); axes_gc[2].axis('off')
                    plt.tight_layout()
                    safe_img_basename = os.path.basename(img_rel_path).rsplit('.', 1)[0].replace('/', '_')
                    grad_cam_filename = f"gradcam_{k_idx}_{safe_img_basename}.png"
                    plt.savefig(os.path.join(gradcam_dir, grad_cam_filename)); plt.close(fig_gc)
                    print(f"Saved Grad-CAM for {img_rel_path} (True: {true_labels_names[k_idx]}, Pred: {preds_names[k_idx]})")
                except Exception as e_grad_loop:
                    print(f"Error generating Grad-CAM for {img_rel_path}: {e_grad_loop}"); traceback.print_exc()

        # 5. Inference Time
        print("\nMeasuring Inference Time...")
        num_inference_samples = min(10, nb_samples_test, len(test_generator.filenames) if test_generator else 0)
        inference_times = []
        if num_inference_samples > 0:
            inference_image_paths = [os.path.join(base_path, test_generator.filenames[i]) for i in range(num_inference_samples)]
            valid_inf_paths_count = 0
            for k_inf_path in inference_image_paths:
                if not os.path.exists(k_inf_path):
                    print(f"Inference img not found: {k_inf_path}")
                    continue
                img_array_inf = get_img_array(k_inf_path, IMAGE_SIZE)
                start_time = time.time()
                _ = model.predict(img_array_inf, verbose=0)
                end_time = time.time()
                inference_times.append(end_time - start_time)
                valid_inf_paths_count += 1
            if inference_times:
                avg_inference_time = np.mean(inference_times)
                print(f"Avg inference time ({valid_inf_paths_count} samples): {avg_inference_time * 1000:.2f} ms, FPS: {1/avg_inference_time:.2f}")
            else:
                print("No valid samples processed for inference time test.")
        else: print("Not enough samples for inference time test.")
    else:
        print("Test generator is effectively empty (nb_samples_test = 0). Evaluation metrics skipped.\n")

print("\nScript finished.")