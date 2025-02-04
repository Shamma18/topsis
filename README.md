# TOPSIS: A Python Implementation

This project provides an implementation of the **Technique for Order Preference by Similarity to Ideal Solution (TOPSIS)**, a widely-used decision-making method. The algorithm evaluates and ranks alternatives based on multiple attributes, considering the importance of each attribute (weights) and its impact type (benefit or cost).

## Features

- **Validation of Inputs**: Ensures proper formatting of input data, weights, and impacts for consistent processing.
- **Comprehensive Error Handling**: Manages issues like missing files, invalid data, and incorrect input parameters.
- **Output Results**: Generates a CSV file that includes original data, calculated scores, and rankings.

---

## How to Run

1. Clone the repository and navigate to the project directory.
2. Ensure Python and the required dependencies are installed.
3. Execute the program using the following command:

   ```bash
   python -m topsis_shamma_102253003.topsis <InputDataSet.csv> <Weights> <Impacts> <Result.csv>
   ```

## Example Usage

- **Input File**: A CSV file containing the dataset where each column represents an attribute, and each row represents an alternative.
- **Weights**: Specify the importance of each attribute as a list (e.g., `[1, 2, 3, 4, 5]`).
- **Impacts**: Define whether each attribute is a benefit (`1`) or a cost (`0`) (e.g., `[1, 0, 1, 1, 0]`).
- **Output File**: A CSV file with calculated TOPSIS scores and rankings.

### Example Weights
```text
[1, 1, 1, 1, 1]
```

### Example Impacts
```text
[1, 0, 1, 0, 1]
```

- `1` indicates a **benefit**
- `0` indicates a **cost**

## Output

The resulting file includes:
- Original data
- Calculated TOPSIS scores
- Rankings based on the scores

## Additional Resources

The package is available on PyPI. You can access it through the following link:

[TOPSIS Package on PyPI](https://pypi.org/project/topsis-Shamma-102253003/#description)
