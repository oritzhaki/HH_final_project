import os
import pandas as pd
import matplotlib.pyplot as plt


# Set the directory containing the CSV files
directory = "dataset/"

# Create a directory for the plots
if not os.path.exists("plots"):
    os.mkdir("plots")

# Loop through all CSV files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        # Load the CSV file as a dataframe
        df = pd.read_csv(os.path.join(directory, filename))

        # Create a line plot of the data
        df.plot()

        # Set the title and axis labels
        plt.title(filename)
        plt.xlabel("X Axis Label")
        plt.ylabel("Y Axis Label")

        # Save the plot as a PNG file in the plots directory
        plt.savefig(os.path.join("plots", f"{filename}.png"))

        # Clear the plot for the next iteration
        plt.clf()
