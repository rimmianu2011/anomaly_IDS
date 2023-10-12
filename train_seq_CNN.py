"""
Project 1 (Anomaly-based IDS)
Group : 3
File_name : train_seq_CNN.py 
@authors : Eshaan Deshpande, Venkat Anurag Nandigala, Anushka Yadav
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# The function 'evaluate_cnn' is used to evaluate a binary classification
# Convolution Neural Network model. It calculates accuracy, precision, recall, 
# f1-score and confusion matrix.
def evaluate_cnn(truth, predicted):
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    for i in range(0, len(predicted)):
        if predicted[i] == 1 and truth[i] == 1:
            true_positives += 1
        if predicted[i] == 0 and truth[i] == 0:
            true_negatives += 1
        if predicted[i] == 1 and truth[i] == 0:
            false_positives += 1
        if predicted[i] == 0 and truth[i] == 1:
            false_negatives += 1
    
    # Calculates accuracy 
    accuracy = (true_positives + true_negatives) / len(y_test)
    
    # Calculates precision
    precision = true_positives / (true_positives + false_positives)
    
    # Calculates recall
    recall = true_positives / (true_positives + false_negatives)
    
    # Calculates F1-score
    f1_score = 2 * (precision * recall) / (precision + recall)

    # confusion matrix is created
    cm = (true_positives, true_negatives, false_positives, false_negatives)
    
    return accuracy, precision, recall, f1_score, cm



# the dataset is read and loaded from a CSV file to a DataFrame.
data = pd.read_csv('final_train_data.csv')

# Extract features (X) and labels (y)
# It is split into X feature and y label. To extract the X part the 
# dataset, 'class' column is dropped so that just the numeric values 
# are taken.
# Then the 'class' column is stored in y where '0' denotes normal and 
# '1' denotes anomaly.
X = data.drop(columns=['class'])  
X = X.astype(float)
y = data['class'].map({'normal': 0, 'anomaly': 1})



model = Sequential()
# This is done so tath the input shape matches the number of columns
model.add(Input(shape=(16, 1)))  # Input shape matches the number of columns

# Added one dimensional Convolutional Layers
model.add(Conv1D(32, 3, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(2))

# Flattened the output from convolutional layers
model.add(Flatten())

# Added Dense Layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Dropout for regularization
model.add(Dense(1, activation='sigmoid'))  # Output layer with 1 neuron and sigmoid activation for binary classification

# The model is compiled
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Dataset Creation
# Split the data into training (70%), validation (15%), and testing (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# Train the model
history = model.fit(X_train, y_train, epochs=30, batch_size=16, validation_data=(X_val, y_val))

y_pred = model.predict(X_val)
y_pred = (y_pred > 0.5).astype(int)
#print("Value")
truth = []
predicted = []
for a in y_val:
    truth.append(a)
for a in y_pred:
    predicted.append(a)


accuracy, precision, recall, f1_score, cm = evaluate_cnn(truth, predicted)

print("Evaluation:-")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1_score:.2f}")
print("Confusion Matrix ( TP, TN, FP, FN)")
print(cm)

# this part saves the model in a file called seq_ids.h5
model.save('seq_ids.h5')
