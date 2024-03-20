import pandas as pd
import os
import pyterrier as pt
from pyterrier.measures import *
import pyltr


def libsvm_to_csv(file_path, save_path=None):
    # Read the libsvm formatted file line by line
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = []
    for index, line in enumerate(lines):
        line, comment = line.strip().split("#")
        line = line.strip().split()
        if any('qid:' in item for item in line):
            label = int(line[0])
            qid_index = [i for i, item in enumerate(line) if 'qid:' in item][0]
            qid = int(line[qid_index].split(':')[1])
            # Split the features into a dictionary
            features = {f'feature_{int(index)}': float(value) for f in line[qid_index + 1:] for index, value in
                        [f.split(':')]}
            # Append a dictionary containing label, query ID, document ID, and features to the data list
            # Here, docids represent indexes in the dataset
            data.append({'label': label, 'qid': qid, 'docid': index, 'comment:': comment, **features})
        else:
            print(f"Warning: 'qid:' not found in line - {line}")

    df = pd.DataFrame(data)
    # Save DataFrame to CSV file if save_path is provided
    if save_path:
        df.to_csv(save_path, index=False)
    return df


def process_directory(input_directory, output_directory):
    # Iterate through all files in the input directory and its subdirectories
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            # Check if the file has a .txt extension
            if file.endswith(".txt"):
                # Construct the full file path
                file_path = os.path.join(root, file)
                # Generate the output file path with .csv extension
                output_file_path = os.path.join(output_directory, file.replace(".txt", ".csv"))
                # Process the file and save the DataFrame
                print("processing file: ", file_path)
                libsvm_to_csv(file_path, save_path=output_file_path)
                #df = read_libsvm(file_path)
                #df.to_csv(save_path, index=False)


if __name__ == '__main__':
    pt.init()
    # convert all libsvm files to csv
    # for i in range(1, 6):
    #     input_directory = output_directory = f'MQ2008/Fold{i}'
    #     # Create the output directory if it doesn't exist
    #     os.makedirs(output_directory, exist_ok=True)
    #     # Process all files in the input directory and its subdirectories
    #     print("processing: " + input_directory)
    #     process_directory(input_directory, output_directory)

    train_path = 'MQ2008/Fold1/train.csv'
    train_df = pd.read_csv(train_path)
    QidLookupTransformer = pt.Transformer.from_df(train_df, uniform=True)
    pipeline = QidLookupTransformer
