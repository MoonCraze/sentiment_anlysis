import os

from preprocessing.preprocess import preprocess_data


def run_preprocessing_news():
    input_path = os.path.join("..", "data", "x_data.json")
    output_path = os.path.join("..", "data", "preprocessed_data_news.json")
    preprocess_data(input_path, output_path)
    return output_path
