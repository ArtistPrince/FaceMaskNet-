import numpy as np
import pandas as pd
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau
import cv2
from tensorflow.keras.optimizers import Adam
import xml.etree.ElementTree as ET
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense, Dropout, Flatten
from sklearn.model_selection import train_test_split
import os
# Function to parse XML files
def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Initialize variables for the image details
    label = None

    # Parse XML to get the label
    for member in root.findall('object'):
        label = member.find('name').text
        # Assuming we only need the label, not bounding box coordinates

    return label
# Paths
annotations_dir = "F:/face/annotations"
images_dir = "F:/face/images"

# Initialize lists to hold data
image_data = []
labels = []

# Loop over each XML file in the annotations folder
for xml_file in os.listdir(annotations_dir):
    if xml_file.endswith('.xml'):
        # Parse the XML to get the label
        xml_path = os.path.join(annotations_dir, xml_file)
        label = parse_xml(xml_path)

        # Corresponding image file
        image_file = xml_file.replace('.xml', '.png')
        image_path = os.path.join(images_dir, image_file)
        
        # Check if the image file exists
        if os.path.exists(image_path):
            # Read and resize image
            image = cv2.imread(image_path)
            image = cv2.resize(image, (128, 128))  # Resize to 128x128

            # Flatten the image for ANN
            image_flat = image.flatten()

            # Append data to lists
            image_data.append(image_flat)
            labels.append(label)

# Convert to numpy arrays
X = np.array(image_data)
y = np.array(labels)
from sklearn.preprocessing import LabelBinarizer

# Encode labels (e.g., 'with_mask' -> 0, 'without_mask' -> 1)
label_encoder = LabelBinarizer()
y_encoded = label_encoder.fit_transform(y)

# Convert y_encoded to a Numpy array 
y_encoded = np.array(y_encoded)

# Convert y_encoded to categorical if it's binary and single-label
if len(y_encoded.shape) == 1:
    print("true")
    y_encoded = to_categorical(y_encoded)

# Function to visualize images with their labels
def visualize_images(X, y_encoded, label_encoder, num_images=10):
    plt.figure(figsize=(15, 10))
    indices = random.sample(range(len(X)), num_images)
    
    for i, idx in enumerate(indices):
        # Reshape the flattened image back to original dimensions
        image = X[idx].reshape(128, 128, 3)
        
        # Get the label index
        label_index = y_encoded[idx]
        
        # Decode the label
        label = label_encoder.inverse_transform([label_index])[0]
        
        # Plot each image in a grid
        plt.subplot(2, 5, i + 1)
        plt.imshow(image.astype('uint8'))
        plt.title(f'Label: {label}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Fit the label encoder and transform labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Call the visualization function
visualize_images(X, y_encoded, label_encoder, num_images=10)

# Normalize pixel values to the range [0, 1]
X_normalized = X / 255.0

# One-hot encode labels for ANN 
y_encoded_one_hot = to_categorical(y_encoded)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_encoded_one_hot, test_size=0.2, random_state=42)

# Check the shape of the training and testing data
print(f"Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")
print(f"Testing data shape: {X_test.shape}, Testing labels shape: {y_test.shape}")

# Build the ANN model
model = Sequential([
    Flatten(input_shape=(49152,)),  # Flatten the 128x128x3 image to a 1D vector
    Dense(256, activation='relu'),  # Hidden layer with 256 neurons
    Dropout(0.5),                   # Dropout layer for regularization
    Dense(128, activation='relu'),  # Hidden layer with 128 neurons
    Dropout(0.5),                   # Dropout layer for regularization
    Dense(3, activation='softmax')  # Output layer for 3 classes (if one-hot encoded)
])

# Compile the model with an optimizer, loss, and evaluation metric
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Display the model's architecture
model.summary()

# Train the ANN model
history = model.fit(
    X_train, y_train,
    epochs=20,                  # Number of epochs to train
    batch_size=32,              # Size of each training batch
    validation_data=(X_test, y_test),  # Use the test set for validation
    verbose=2                   # Verbose output
)

'''# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Plot learning curves
plt.figure(figsize=(14, 6))

# Plot training & validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()'''

# Build an improved ANN model with reduced regularization
improved_model = Sequential([
    Flatten(input_shape=(49152,)),  # Flatten the 128x128x3 image to a 1D vector
    
    # First Dense Block
    Dense(1024, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),  # Reduced dropout rate
    
    # Second Dense Block
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    # Third Dense Block
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    # Fourth Dense Block
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    # Output Layer for 3 classes
    Dense(3, activation='softmax')
])

from tensorflow.keras.optimizers import Adam

# Adjusted learning rate
improved_optimizer = Adam(learning_rate=0.0005)

# Compile the improved model
improved_model.compile(optimizer=improved_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Display the improved model summary
improved_model.summary()

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Define Early Stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',        # Monitor the validation loss
    patience=10,               # Stop if no improvement after 10 epochs
    restore_best_weights=True  # Restore the best weights after stopping
)

# Define Learning Rate Scheduler
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',  # Monitor validation loss
    factor=0.5,           # Reduce the learning rate by a factor of 0.5
    patience=3,           # Wait for 3 epochs without improvement
    verbose=1,            # Verbosity mode
    min_lr=1e-6           # Minimum learning rate
)

# Train the improved model
history_improved = improved_model.fit(
    X_train, y_train,
    epochs=20,  # Increased epochs
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, lr_scheduler],  # Include early stopping and learning rate scheduler
    verbose=2
)
'''
# Evaluate the improved model on the test set
test_loss, test_accuracy = improved_model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy for Improved Model: {test_accuracy:.2f}")

# Plot learning curves for the improved model
plt.figure(figsize=(14, 6))

# Plot training & validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history_improved.history['accuracy'], label='Training Accuracy')
plt.plot(history_improved.history['val_accuracy'], label='Validation Accuracy')
plt.title('Improved Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history_improved.history['loss'], label='Training Loss')
plt.plot(history_improved.history['val_loss'], label='Validation Loss')
plt.title('Improved Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()
'''
# Reshape the input data to the original image shape (128x128x3)
X_train_reshaped = X_train.reshape(-1, 128, 128, 3)
X_test_reshaped = X_test.reshape(-1, 128, 128, 3)

# Build a simple CNN model
cnn_model = Sequential([
    # First Convolutional Block
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    # Second Convolutional Block
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    # Third Convolutional Block
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.4),
    
    # Flatten and Dense Layers
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(3, activation='softmax')  # Output layer for 3 classes
])

# Compile the CNN model
cnn_optimizer = Adam(learning_rate=0.0001)
cnn_model.compile(optimizer=cnn_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Display the CNN model summary
cnn_model.summary()

# Train the CNN model with Early Stopping
history_cnn = cnn_model.fit(
    X_train_reshaped, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test_reshaped, y_test),
    callbacks=[early_stopping, lr_scheduler],  # Include early stopping and learning rate scheduler
    verbose=2
)
'''
# Evaluate the CNN model on the test set
test_loss, test_accuracy = cnn_model.evaluate(X_test_reshaped, y_test, verbose=0)
print(f"Test Accuracy for CNN Model: {test_accuracy:.2f}")

# Plot learning curves for the CNN model
plt.figure(figsize=(14, 6))

# Plot training & validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history_cnn.history['accuracy'], label='Training Accuracy')
plt.plot(history_cnn.history['val_accuracy'], label='Validation Accuracy')
plt.title('CNN Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history_cnn.history['loss'], label='Training Loss')
plt.plot(history_cnn.history['val_loss'], label='Validation Loss')
plt.title('CNN Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()'''

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

# Load MobileNetV2 with pre-trained weights, excluding the top classification layer
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Freeze the base model layers
base_model.trainable = False

# Add custom top layers for your specific dataset
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(3, activation='softmax')(x)

# Combine the base model with custom layers
transfer_model = Model(inputs=base_model.input, outputs=predictions)

# Compile the transfer learning model
transfer_optimizer = Adam(learning_rate=0.0001)
transfer_model.compile(optimizer=transfer_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Display the transfer learning model summary
transfer_model.summary()

# Train the transfer learning model
history_transfer = transfer_model.fit(
    X_train_reshaped, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test_reshaped, y_test),
    callbacks=[early_stopping, lr_scheduler],  # Include early stopping and learning rate scheduler
    verbose=2
)
'''
# Evaluate the transfer learning model on the test set
test_loss, test_accuracy = transfer_model.evaluate(X_test_reshaped, y_test, verbose=0)
print(f"Test Accuracy for Transfer Learning Model: {test_accuracy:.2f}")

# Plot learning curves for the transfer learning model
plt.figure(figsize=(14, 6))

# Plot training & validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history_transfer.history['accuracy'], label='Training Accuracy')
plt.plot(history_transfer.history['val_accuracy'], label='Validation Accuracy')
plt.title('Transfer Learning Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history_transfer.history['loss'], label='Training Loss')
plt.plot(history_transfer.history['val_loss'], label='Validation Loss')
plt.title('Transfer Learning Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()


# Function to visualize predictions from the transfer learning model
def visualize_predictions(model, X_data, y_true, label_encoder, num_images=10):
    plt.figure(figsize=(15, 10))
    
    # Choose random indices to display
    indices = np.random.choice(range(len(X_data)), num_images, replace=False)
    
    for i, idx in enumerate(indices):
        # Get the image and true label
        image = X_data[idx]
        
        # If the image is normalized (0-1), rescale it back to 0-255
        if image.max() <= 1.0:
            image = (image * 255).astype('uint8')
        
        # If the image is flattened, reshape it back to (128, 128, 3)
        if image.shape != (128, 128, 3):
            image = image.reshape(128, 128, 3)
        
        # Ensure the image data type is uint8 for visualization
        if image.dtype != 'uint8':
            image = image.astype('uint8')
        
        # Get the true label
        true_label = label_encoder.inverse_transform([y_true[idx].argmax()])[0]
        
        # Predict the label using the transfer learning model
        prediction = model.predict(np.expand_dims(image / 255.0, axis=0))  # Normalize image for prediction
        predicted_label = label_encoder.inverse_transform([prediction.argmax()])[0]
        
        # Plot each image in a grid
        plt.subplot(2, 5, i + 1)
        plt.imshow(image)
        plt.title(f'True: {true_label}\nPred: {predicted_label}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Call the visualization function using the correct model name 'transfer_model'
visualize_predictions(transfer_model, X_test_reshaped, y_test, label_encoder, num_images=10)
'''
'''
# Assuming the three models are loaded
try:
  ann_model = tf.keras.models.load_model(improved_model)
  print("ann_model loaded successfully.")
except OSError:
  print("Could not load ann_model from improved_model.h5. Trying to load as a SavedModel.")
  try:
    ann_model = tf.keras.layers.TFSMLayer("improved_model", call_endpoint='serving_default')
    print("ann_model loaded successfully as a SavedModel.")
  except Exception as e:
    print(f"Error loading ann_model as a SavedModel: {e}")
    ann_model = None

try:
  cnn_model = tf.keras.models.load_model(cnn_model.h5)
  print("cnn_model loaded successfully.")
except OSError:
  print("Could not load cnn_model from cnn_model.h5. Trying to load as a SavedModel.")
  try:
    cnn_model = tf.keras.layers.TFSMLayer("cnn_model", call_endpoint='serving_default')
    print("cnn_model loaded successfully as a SavedModel.")
  except Exception as e:
    print(f"Error loading cnn_model as a SavedModel: {e}")
    cnn_model = None

try:
  transfer_model = tf.keras.models.load_model(transfer_model.h5)
  print("transfer_model loaded successfully.")
except OSError:
  print("Could not load transfer_model from transfer_model.h5. Trying to load as a SavedModel.")
  try:
    transfer_model = tf.keras.layers.TFSMLayer("transfer_model", call_endpoint='serving_default')
    print("transfer_model loaded successfully as a SavedModel.")
  except Exception as e:
    print(f"Error loading transfer_model as a SavedModel: {e}")
    transfer_model = None
'''

def ensemble_predict(image):
    # Preprocess the input image
    image_resized = cv2.resize(image, (128, 128))
    image_normalized = image_resized / 255.0
    image_reshaped = np.expand_dims(image_normalized, axis=0)
    
    # Get predictions from each model
    ann_prediction = improved_model.predict(image_reshaped.flatten().reshape(1, -1))
    cnn_prediction = cnn_model.predict(image_reshaped)
    transfer_prediction = transfer_model.predict(image_reshaped)
    
    # Combine predictions (Weighted Averaging)
    final_prediction = (0.3 * ann_prediction) + (0.3 * cnn_prediction) + (0.4 * transfer_prediction)
    
    # Get the final class label
    label_index = np.argmax(final_prediction)
    label = labels[label_index]
    confidence = final_prediction[0][label_index] * 100
    
    return label, confidence

# Example usage with real-time detection
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    
    # Assuming you have a face detection step
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        label, confidence = ensemble_predict(face)
        
        # Draw bounding box and prediction
        color = (0, 255, 0) if label == 'with_mask' else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{label}: {confidence:.2f}%", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    cv2.imshow("Ensemble Face Mask Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
