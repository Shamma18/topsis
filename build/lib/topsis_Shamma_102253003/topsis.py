import sys
import pandas as pd
import numpy as np
import os

def topsis_calculation(data_matrix, weight_vector, impact_vector):
    data_matrix = np.array(data_matrix)
    norm_matrix = data_matrix / np.sqrt((data_matrix**2).sum(axis=0))
    weighted_matrix = norm_matrix * weight_vector

    ideal_best = np.where(impact_vector == 1, np.max(weighted_matrix, axis=0), np.min(weighted_matrix, axis=0))
    ideal_worst = np.where(impact_vector == 1, np.min(weighted_matrix, axis=0), np.max(weighted_matrix, axis=0))

    distance_best = np.sqrt(((weighted_matrix - ideal_best)**2).sum(axis=1))
    distance_worst = np.sqrt(((weighted_matrix - ideal_worst)**2).sum(axis=1))

    performance_scores = distance_worst / (distance_best + distance_worst)
    rankings = performance_scores.argsort()[::-1] + 1
    return performance_scores, rankings

def main():
    if len(sys.argv) != 5:
        print("Usage: python 102253003.py <InputDataSet.csv> <Weights> <Impacts> <Result.csv>")
        sys.exit(1)

    input_path = sys.argv[1]
    weight_input = sys.argv[2]
    impact_input = sys.argv[3]
    output_path = sys.argv[4]

    if not os.path.isfile(input_path):
        print(f"Error: File '{input_path}' not found.")
        sys.exit(1)

    try:
        dataset = pd.read_csv(input_path)
        if dataset.shape[1] < 3:
            print("Error: Input file must have at least three columns.")
            sys.exit(1)
        criteria_matrix = dataset.iloc[:, 1:].values
    except Exception as err:
        print(f"Error reading the input file: {err}")
        sys.exit(1)

    try:
        weight_list = list(map(float, weight_input.split(',')))
        impact_list = list(map(float, impact_input.split(',')))
        if not all(impact in [1, -1] for impact in impact_list):
            print("Error: Impacts must be 1 (benefit) or -1 (cost).")
            sys.exit(1)
    except ValueError:
        print("Error: Weights must be numbers and impacts must be 1 or -1, separated by commas.")
        sys.exit(1)

    if len(weight_list) != criteria_matrix.shape[1] or len(impact_list) != criteria_matrix.shape[1]:
        print("Error: Number of weights and impacts must match the number of criteria.")
        sys.exit(1)

    if not np.issubdtype(criteria_matrix.dtype, np.number):
        print("Error: Criteria columns must contain only numeric values.")
        sys.exit(1)

    scores, ranks = topsis_calculation(criteria_matrix, np.array(weight_list), np.array(impact_list))

    try:
        dataset["Score"] = scores
        dataset["Rank"] = ranks
        dataset.to_csv(output_path, index=False)
        print(f"Results successfully saved to {output_path}")
    except Exception as err:
        print(f"Error writing results to file: {err}")
        sys.exit(1)

if __name__ == "__main__":
    main()
