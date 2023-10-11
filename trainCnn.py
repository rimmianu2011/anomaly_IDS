import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


normal_dir = 'anomaly/'
anomaly_dir = 'normal/'


X = []
y = []

normal_files = os.listdir(normal_dir)
for filename in normal_files:
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Add more extensions if needed
        img = Image.open(os.path.join(normal_dir, filename))
        img = img.resize((64, 64))  # Resize the image to your desired dimensions
        img = np.array(img)  # Convert the PIL image to a NumPy array
        X.append(img)
        y.append(0)  # 0 for normal images


anomaly_files = os.listdir(anomaly_dir)
for filename in anomaly_files:
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Add more extensions if needed
        img = Image.open(os.path.join(anomaly_dir, filename))
        img = img.resize((64, 64))  # Resize the image to your desired dimensions
        img = np.array(img)  # Convert the PIL image to a NumPy array
        X.append(img)
        y.append(1)  # 1 for anomaly images
        
X = np.array(X)
y = np.array(y)

X = X.reshape(-1, 64, 64, 1)


# Define the CNN model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),  # Updated input shape
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')  # 2 classes (normal and anomaly)
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Data preprocessing (assuming you have the data loaded in 'X' and 'y' as NumPy arrays)
# X should be normalized to the range [0, 1]
#X = X / 255.0

# Split the dataset into training and validation sets
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

model.save('ids.h5')

# Evaluate the model
#test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')



