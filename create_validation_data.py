import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

dataset_name = 'Test_data.csv'

# Load your CSV file into a DataFrame (replace 'data.csv' with your file path)
df = pd.read_csv(dataset_name)

# Exclude non-numeric columns (assuming 'class' is the target column)
numeric_df = df.select_dtypes(include=[np.number])

#numeric_df['class'] = df['class']

# Convert 'class' column to 0 for 'normal' and 1 for 'anomaly'
numeric_df['class'] = df['class'].replace({'normal': 0, 'anomaly': 1})

print(numeric_df.head())


# Select the specified columns
selected_columns = ['class', 'same_srv_rate', 'dst_host_srv_count',
       'dst_host_same_srv_rate', 'logged_in', 'dst_host_srv_serror_rate',
       'dst_host_serror_rate', 'serror_rate', 'srv_serror_rate', 'count',
       'dst_host_count', 'dst_host_srv_rerror_rate', 'rerror_rate',
       'dst_host_rerror_rate', 'srv_rerror_rate', 'dst_host_diff_srv_rate',
       'diff_srv_rate']

# Create a new DataFrame containing only the selected columns
new_df = df[selected_columns]

# Separate the 'class' column and the numeric columns
class_column = df['class'].replace({'normal': 0, 'anomaly': 1})
numeric_columns = new_df.drop(columns=['class'])

# Initialize the Min-Max scaler
scaler = MinMaxScaler()

# Normalize the numeric columns
normalized_numeric_columns = pd.DataFrame(scaler.fit_transform(numeric_columns), columns=numeric_columns.columns)

# Combine the 'class' column with the normalized numeric columns
normalized_df = pd.concat([class_column, normalized_numeric_columns], axis=1)

# Save the normalized dataset to a different CSV file (e.g., 'normalized_data.csv')
normalized_df.to_csv('final_test_data.csv', index=False)

# Save the new DataFrame to a different dataset (e.g., 'selected_data.csv')
#new_df.to_csv('final_train_data.csv', index=False)

# Save the selected numeric features to a new CSV file (if needed)
#selected_features_df.to_csv('selected_numeric_features.csv', index=False)

