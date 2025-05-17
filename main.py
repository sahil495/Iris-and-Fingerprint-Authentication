# by concatinating both here is our multi_model
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

# Image dimensions and number of classes
IMG_HEIGHT = 128
IMG_WIDTH = 128
NUM_CLASSES = 45

# Path to the folder containing fingerprints and eye images
fingerprint_image_folder = 'Iris_and_Fingerprint_data_50_sample_per_class/fp50_data'   # Modify with correct path
eye_image_folder = 'Iris_and_Fingerprint_data_50_sample_per_class/eye50_data'    # Modify with correct path

def load_images(folder, num_classes):
    images = []
    labels = []
    image_names = []

    image_files = sorted([f for f in os.listdir(folder) if f.endswith('.bmp')],
                         key=lambda x: int(x.split('.')[0]))

    for img_file in image_files:
        img_path = os.path.join(folder, img_file)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            images.append(img)
            image_names.append(img_file)
            img_num = int(img_file.split('.')[0])
            label = (img_num - 1) // 50  # Label assignment logic
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    images = images / 255.0  # Normalize
    labels = to_categorical(labels, num_classes)  # One-hot encode labels

    return images, labels, image_names

# Load eye and fingerprint images
X_eyes, y_eyes, eye_image_names = load_images(eye_image_folder, NUM_CLASSES)
X_fps, y_fps, fp_image_names = load_images(fingerprint_image_folder, NUM_CLASSES)



# Function to decode one-hot encoded labels
def decode_labels(one_hot_labels):
    return np.argmax(one_hot_labels, axis=1)

# Function to show random images from both datasets along with names and labels
def show_random_images(images_eyes, images_fps, eye_names, fp_names, eye_labels, fp_labels, num_images=10):
    indices_eyes = np.random.choice(len(images_eyes), num_images, replace=False)
    indices_fps = np.random.choice(len(images_fps), num_images, replace=False)

    plt.figure(figsize=(12, 12))

    # Plot Eye Images
    for i, idx in enumerate(indices_eyes):
        plt.subplot(4, 5, i + 1)
        plt.imshow(images_eyes[idx])
        plt.axis('off')
        plt.title(f"Eye: {eye_names[idx]}\nLabel: {eye_labels[idx]}")  # Display name and label

    # Plot Fingerprint Images
    for i, idx in enumerate(indices_fps):
        plt.subplot(4, 5, i + 11)
        plt.imshow(images_fps[idx])
        plt.axis('off')
        plt.title(f"FP: {fp_names[idx]}\nLabel: {fp_labels[idx]}")  # Display name and label

    plt.tight_layout()
    plt.show()

# Decode labels if they are one-hot encoded
y_eyes_decoded = decode_labels(y_eyes) if len(y_eyes.shape) > 1 else y_eyes
y_fps_decoded = decode_labels(y_fps) if len(y_fps.shape) > 1 else y_fps

# Call the function to display random images from both datasets with decoded labels
show_random_images(X_eyes, X_fps, eye_image_names, fp_image_names, y_eyes_decoded, y_fps_decoded, num_images=10)


# Split data into training and testing sets
X_train_eyes, X_test_eyes, y_train, y_test, train_eye_names, test_eye_names = train_test_split(
    X_eyes, y_eyes, eye_image_names, test_size=0.2, random_state=42)

X_train_fps, X_test_fps, _, _, train_fp_names, test_fp_names = train_test_split(
    X_fps, y_fps, fp_image_names, test_size=0.2, random_state=42)


def feature_extractor(input_shape):
    model = Sequential()
    model.add(Conv2D(128, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.45))

    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())  # Output the feature map
    return model


input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)

# Define two inputs: one for eyes and one for fingerprints
input_eyes = Input(shape=input_shape)
input_fps = Input(shape=input_shape)

# Feature extraction for both
features_eyes = feature_extractor(input_shape)(input_eyes)
features_fps = feature_extractor(input_shape)(input_fps)

# Concatenate the features extracted from both image types
concatenated_features = concatenate([features_eyes, features_fps])

# Fully connected layer and output

output = Dense(1024, activation='relu')(concatenated_features)
output = Dropout(0.5)(output)

output = Dense(NUM_CLASSES, activation='softmax')(concatenated_features)

# Define the full model with two inputs
model = Model(inputs=[input_eyes, input_fps], outputs=output)

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])


history = model.fit([X_train_eyes, X_train_fps], y_train,
                    validation_data=([X_test_eyes, X_test_fps], y_test),
                    epochs=50,  # Adjust as needed
                    batch_size=64)  # Adjust as needed

def plot_performance(history):
    # Plot accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss')

    plt.show()

# Plot training and validation performance
plot_performance(history)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate([X_test_eyes, X_test_fps], y_test)
print(f"Test Accuracy: {test_acc:.4f}")


# Predict the labels for the test set
y_pred = model.predict([X_test_eyes, X_test_fps])
y_pred_labels = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Show classification report
print(classification_report(y_test_labels, y_pred_labels))


# Function to show test images along with actual and predicted labels
def show_test_images_with_predictions(images_eyes, images_fps, eye_names, fp_names, actual_eye_labels, actual_fp_labels, predicted_labels, num_images=10):
    indices_eyes = np.random.choice(len(images_eyes), num_images, replace=False)
    indices_fps = np.random.choice(len(images_fps), num_images, replace=False)

    plt.figure(figsize=(12, 12))

    # Plot Eye Images with actual and predicted labels
    for i, idx in enumerate(indices_eyes):
        plt.subplot(4, 5, i + 1)
        plt.imshow(images_eyes[idx])
        plt.axis('off')
        plt.title(f"Eye: {eye_names[idx]}\nActual: {actual_eye_labels[idx]}\nPred: {predicted_labels[idx]}")  # Display name, actual, and predicted labels

    # Plot Fingerprint Images with actual and predicted labels
    for i, idx in enumerate(indices_fps):
        plt.subplot(4, 5, i + 11)
        plt.imshow(images_fps[idx])
        plt.axis('off')
        plt.title(f"FP: {fp_names[idx]}\nActual: {actual_fp_labels[idx]}\nPred: {predicted_labels[idx]}")  # Display name, actual, and predicted labels

    plt.tight_layout()
    plt.show()

# Decode labels if they are one-hot encoded
y_test_decoded = decode_labels(y_test) if len(y_test.shape) > 1 else y_test
y_pred_decoded = decode_labels(y_pred) if len(y_pred.shape) > 1 else y_pred

# Show test images along with actual and predicted labels
show_test_images_with_predictions(X_test_eyes, X_test_fps, test_eye_names, test_fp_names, y_test_decoded, y_test_decoded, y_pred_decoded, num_images=10)
