"""
Project 1 (Anomaly-based IDS)
Group : 3
File_name : preProcess.py 
@authors : Eshaan Deshpande, Venkat Anurag Nandigala, Anushka Yadav
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Train_data.csv is the data taken from web and processed to help 
# optimize the model training and testing.
dataset_name = 'Train_data.csv'

# Reads the data using panda into a DataFrame
df = pd.read_csv(dataset_name)


# This is done to exclude the columns which have non-numeric values and the columns 
# that have no values.
numeric_df = df.select_dtypes(include=[np.number])

# Converted 'class' column to contain only 0 and 1, where 0 denotes
# 'normal' and 1 denotes 'anomaly'.
numeric_df['class'] = df['class'].replace({'normal': 0, 'anomaly': 1})

print(numeric_df.head())

# Calculate the correlation matrix for numeric features
correlation_matrix = numeric_df.corr()

# This part gets the absolute values of correlations for each feature that has 
# the target as 'class'.
correlation_with_target = abs(correlation_matrix['class'])

# This part sorts the features based on their correlation with the target in 
# descending order by setting 'ascending=False' and then takes 16 features and 
# the target variable 'class' from the sorted list.
top_features = correlation_with_target.sort_values(ascending=False)[:17].index

# Select the top 16 features from the DataFrame.
selected_features_df = numeric_df[top_features]

# Outputs the feature correlation matrix for numeric features.
print("Feature Correlation Matrix:")
print(correlation_matrix)

# Prints the name of the first 17 numeric features.
print("\nTop 17 Numeric Features:")
print(top_features)

selected_columns = top_features

# Created a new DataFrame that contains the first 17 selected 
# columns.
new_df = df[selected_columns]

# This part separates the 'class' column from the numeric 
# columns. This is done by first extracting the 'class' column
# and then dropping that column to extract the numeric values.
class_column = new_df['class']
numeric_columns = new_df.drop(columns=['class'])

# Initialized the Min-Max scaler for scaling the values.
scaler = MinMaxScaler()

# Normalize the numeric columns
normalized_numeric_columns = pd.DataFrame(scaler.fit_transform(numeric_columns), columns=numeric_columns.columns)

# After the numeric values are normalized it is combined with the
# 'class' column.
normalized_df = pd.concat([class_column, normalized_numeric_columns], axis=1)

# The normalized dataset is then saved to a new file named 'final_train_data.csv'.
normalized_df.to_csv('final_train_data.csv', index=False)

plt.figure(figsize=(17, 17))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5)

# The heatmap is saved as an image file.
plt.savefig('correlation_matrix.png')

# Displays the plot
plt.show()
