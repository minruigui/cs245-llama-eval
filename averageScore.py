import os
import pandas as pd

folder_path = '/home/ubuntu/projects/results/results_davinci'  # Replace with your folder path

cors = []

# Iterate through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)

            # Check if 'davinci_correct' column exists
            if 'davinci_correct' in df.columns:
                # Count the 'True' values and total values
                cor = df['davinci_correct'].sum()/len(df['davinci_correct'])
                print(filename.split('.csv')[0], )
                cors.append(cor)
        except Exception as e:
            print(f"Error processing file {filename}: {e}")
import numpy as np

print(f"Percentage of 'True' values: {np.mean(cors):.3f}")

