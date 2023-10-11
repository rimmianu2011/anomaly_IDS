import pandas as pd
import random

# Load your CSV file into a DataFrame (replace 'data.csv' with your file path)
df = pd.read_csv('Train_data.csv')

# Separate the data into 'normal' and 'anomaly' classes
normal_data = df[df['class'] == 'normal']
anomaly_data = df[df['class'] == 'anomaly']

# Randomly select 1000 records from 'normal' class
random_normal_selection = normal_data.sample(n=10000, random_state=42)

# Randomly select 500 records from 'anomaly' class
random_anomaly_selection = anomaly_data.sample(n=7000, random_state=42)

# Combine the selected records into a new DataFrame
selected_data = pd.concat([random_normal_selection, random_anomaly_selection])

# Save the selected data to a new CSV file (e.g., 'selected_data.csv')
selected_data.to_csv('cleaned_data.csv', index=False)
