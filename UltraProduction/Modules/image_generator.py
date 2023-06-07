import matplotlib.pyplot as plt
import os
import pandas as pd

def plot_csv_files(data_dir):
    # Iterate over each directory in the ConductivityData directory
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            # Check if file is a _graph.csv file
            if file.endswith("_graph.csv"):
                csv_file_path = os.path.join(root, file)
                
                # Load the CSV file as a dataframe
                df = pd.read_csv(csv_file_path)

                # Create a line plot of the data
                df.plot()

                # Set the title and axis labels
                plt.title(file)
                plt.xlabel("X Axis Label")
                plt.ylabel("Y Axis Label")

                # Save the plot as a PNG file in the directory of the CSV file
                img_file_path = os.path.join(root, file.replace('_graph.csv', '_img.png'))
                plt.savefig(img_file_path)

                # Print the path of the saved image
                print(f"Saved image at {img_file_path}")

                # Clear the plot for the next iteration
                plt.clf()

def run():
    data_dir = "ConductivityData"
    plot_csv_files(data_dir)
