#4
import pandas as pd

def find_s_algorithm(file_path):
    data = pd.read_csv(file_path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].str.lower()

    hypothesis = None
    for i in range(len(y)):
        if y[i] == 'yes':
            if hypothesis is None:
                hypothesis = X[i].copy()
            else:
                hypothesis = ['?' if hypothesis[j] != X[i][j] else hypothesis[j] for j in range(len(hypothesis))]
    return data, hypothesis

if __name__ == "__main__":
    path = "enjoysport.csv"
    data, hypothesis = find_s_algorithm(path)

    print("Training Data:")
    print(data)

    print("\nMost Specific Hypothesis:")
    print(hypothesis)
