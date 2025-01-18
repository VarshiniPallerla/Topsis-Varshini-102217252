# TOPSIS Python Package

Made By: **Varshini Pallerla**  
Roll No: **102217252**

## Description

The TOPSIS (Technique for Order Preference by Similarity to Ideal Solution) is a multi-criteria decision-making method used to rank alternatives based on their similarity to an ideal solution. The TOPSIS Python Package is a library that provides an easy-to-use implementation of this method, helps in selecting the best alternative from a set of alternatives based on their performance across multiple criteria.

## Installation

You can install the package using pip:
pip install topsis-varshinipallerla

## Usage

Use the following command to perform TOPSIS analysis on a dataset:

python topsis.py data.csv "1,0,1,0,1" "+,-,+,-,+"

- topsis.py: Python file with code
- data.csv: Path to the input CSV file containing the dataset.
- "1,0,1,0,1": Weights for each criterion separated by commas.
- "+,-,+,-,+": Impacts for each criterion (either + for maximizing or - for minimizing).

The output will be saved in a file named output.csv
