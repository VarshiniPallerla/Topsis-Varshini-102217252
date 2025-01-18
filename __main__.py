import math
import pandas as pd
import sys
import numpy as np

def normalize_dataset(dataset):
    """Normalize each column of the dataset."""
    return dataset.apply(lambda col: col / np.sqrt((col ** 2).sum()), axis=0)

def apply_weights(dataset, weights):
    """Apply the weights to the dataset."""
    return dataset.multiply(weights, axis=1)

def calculate_ideal_values(dataset, impacts):
    """Calculate the ideal positive and negative values."""
    vpos = dataset.max(axis=0) * (np.array(impacts) == '+') + dataset.min(axis=0) * (np.array(impacts) == '-')
    vneg = dataset.min(axis=0) * (np.array(impacts) == '+') + dataset.max(axis=0) * (np.array(impacts) == '-')
    return vpos, vneg

def calculate_performance_scores(dataset, vpos, vneg):
    """Calculate the performance scores."""
    euclidean_distance_positive = np.sqrt(((dataset - vpos) ** 2).sum(axis=1))
    euclidean_distance_negative = np.sqrt(((dataset - vneg) ** 2).sum(axis=1))
    scores = euclidean_distance_negative / (euclidean_distance_positive + euclidean_distance_negative)
    return scores

def topsis(arglist):
    try:
        # Read the dataset
        dataset = pd.read_csv(arglist[0])
        identifiers = dataset.iloc[:, 0]
        dataset = dataset.iloc[:, 1:]

        # Normalize the dataset
        dataset = normalize_dataset(dataset)

        # Parse and normalize weights
        weights = list(map(float, arglist[1].split(',')))
        if len(weights) != len(dataset.columns):
            raise ValueError(f"Number of weights ({len(weights)}) must match the number of criteria ({len(dataset.columns)}).")
        weights = np.array(weights) / sum(weights)

        # Apply weights
        dataset = apply_weights(dataset, weights)

        # Parse impacts
        impacts = arglist[2].split(',')
        if len(impacts) != len(dataset.columns):
            raise ValueError(f"Number of impacts ({len(impacts)}) must match the number of criteria ({len(dataset.columns)}).")
        if not all(impact in ['+', '-'] for impact in impacts):
            raise ValueError("Impacts must be either '+' or '-'.")

        # Calculate ideal values
        vpos, vneg = calculate_ideal_values(dataset, impacts)

        # Calculate performance scores
        scores = calculate_performance_scores(dataset, vpos, vneg)
        dataset['Performance Score'] = scores
        dataset['Rank'] = scores.rank(ascending=False).astype(int)

        # Add back identifiers and save results
        dataset.insert(0, "Identifier", identifiers)
        dataset.to_csv(arglist[3], index=False)
        print(f"TOPSIS analysis completed. Results saved to '{arglist[3]}'.")
    
    except FileNotFoundError:
        print(f"Error: File '{arglist[0]}' not found.")
    except ValueError as ve:
        print(f"Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Error: Incorrect number of parameters. Expected: <InputDataFile> <Weights> <Impacts> <ResultFileName>.")
        sys.exit(1)

    topsis(sys.argv[1:])
