import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

dataset_name = 'Train_data.csv'

# Load your CSV file into a DataFrame (replace 'data.csv' with your file path)
df = pd.read_csv(dataset_name)


# Exclude non-numeric columns (assuming 'class' is the target column)
numeric_df = df.select_dtypes(include=[np.number])

#numeric_df['class'] = df['class']

# Convert 'class' column to 0 for 'normal' and 1 for 'anomaly'
numeric_df['class'] = df['class'].replace({'normal': 0, 'anomaly': 1})

print(numeric_df.head())

# Calculate the correlation matrix for numeric features
correlation_matrix = numeric_df.corr()

# Get the absolute values of correlations for each feature with the target ('class')
correlation_with_target = abs(correlation_matrix['class'])

# Sort the features by their correlation with the target in descending order
top_features = correlation_with_target.sort_values(ascending=False)[:17].index

# Select the top 16 features from the DataFrame
selected_features_df = numeric_df[top_features]

# Output the feature correlation matrix for numeric features
print("Feature Correlation Matrix:")
print(correlation_matrix)

# Output the names of the top 17 features
print("\nTop 17 Numeric Features:")
print(top_features)

# Select the specified columns
selected_columns = top_features

# Create a new DataFrame containing only the selected columns
new_df = df[selected_columns]

# Separate the 'class' column and the numeric columns
class_column = new_df['class']
numeric_columns = new_df.drop(columns=['class'])

# Initialize the Min-Max scaler
scaler = MinMaxScaler()

# Normalize the numeric columns
normalized_numeric_columns = pd.DataFrame(scaler.fit_transform(numeric_columns), columns=numeric_columns.columns)

# Combine the 'class' column with the normalized numeric columns
normalized_df = pd.concat([class_column, normalized_numeric_columns], axis=1)

# Save the normalized dataset to a different CSV file (e.g., 'normalized_data.csv')
normalized_df.to_csv('final_train_data.csv', index=False)

# Save the new DataFrame to a different dataset (e.g., 'selected_data.csv')
#new_df.to_csv('final_train_data.csv', index=False)

# Save the selected numeric features to a new CSV file (if needed)
#selected_features_df.to_csv('selected_numeric_features.csv', index=False)

plt.figure(figsize=(17, 17))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5)

# Save the heatmap as an image file (e.g., PNG)
plt.savefig('correlation_matrix.png')

# Show the plot (optional)
plt.show()
