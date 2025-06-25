import os

from preprocessing.preprocess import preprocess_data

if __name__ == "__main__":
    input_path = os.path.join("..", "data", "x_data.json")
    output_path = os.path.join("..", "data", "preprocessed_data1.json")
    preprocess_data(input_path, output_path)
    print(f"Preprocessed data saved to {output_path}")
