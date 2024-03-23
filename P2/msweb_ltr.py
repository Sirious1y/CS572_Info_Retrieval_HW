import pandas as pd
import numpy as np
import os
import pyterrier as pt
from pyterrier.measures import *
import pyltr
import xgboost as xgb
import fastrank


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
            features = [float(value) for f in line[qid_index + 1:] for index, value in
                        [f.split(':')]]
            # Append a dictionary containing label, query ID, document ID, and features to the data list
            # Here, docids represent indexes in the dataset
            data.append({'label': label, 'qid': qid, 'docid': index, 'comment:': comment, 'features': features})
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
    pd.set_option('display.max_columns', None)
    # convert all libsvm files to csv
    for i in range(1, 6):
        input_directory = output_directory = f'MQ2008/Fold{i}'
        # Create the output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)
        # Process all files in the input directory and its subdirectories
        print("processing: " + input_directory)
        process_directory(input_directory, output_directory)

    # variables to store performances
    lmart_x_feature_importance = None
    fr_feature_importance = None
    lmart_x_ndcg = 0
    fr_ndcg = 0
    lmart_x_map = 0
    fr_map = 0
    for i in range(1, 6):
        train_path = f'MQ2008/Fold{i}/train.csv'
        val_path = f'MQ2008/Fold{i}/vali.csv'
        test_path = f'MQ2008/Fold{i}/test.csv'
        # load all csv files, convert features to list
        train_df = pd.read_csv(train_path, converters={'features': pd.eval})
        val_df = pd.read_csv(val_path, converters={'features': pd.eval})
        test_df = pd.read_csv(test_path, converters={'features': pd.eval})
        # convert features to nparray
        train_df['features'] = train_df['features'].apply(lambda x: np.array(x))
        val_df['features'] = val_df['features'].apply(lambda x: np.array(x))
        test_df['features'] = test_df['features'].apply(lambda x: np.array(x))
        # convert qid to str
        train_df['qid'] = train_df['qid'].apply(lambda x: str(x))
        val_df['qid'] = val_df['qid'].apply(lambda x: str(x))
        test_df['qid'] = test_df['qid'].apply(lambda x: str(x))
        # convert label to float64
        train_df['label'] = train_df['label'].astype('float64')
        val_df['label'] = val_df['label'].astype('float64')
        test_df['label'] = test_df['label'].astype('float64')
        # change docid to docno
        train_df = train_df.rename(columns={'docid': 'docno'})
        val_df = val_df.rename(columns={'docid': 'docno'})
        test_df = test_df.rename(columns={'docid': 'docno'})
        # convert docno to string
        train_df['docno'] = train_df['docno'].apply(lambda x: str(x))
        val_df['docno'] = val_df['docno'].apply(lambda x: str(x))
        test_df['docno'] = test_df['docno'].apply(lambda x: str(x))

        # extract topics and qrels
        train_topics = train_df[['qid']].drop_duplicates().reset_index(drop=True)
        train_topics['query'] = train_df['qid'].apply(lambda x: str(x))
        train_qrels = train_df[['qid', 'docno', 'label']].drop_duplicates().reset_index(drop=True)


        val_topics = val_df[['qid']].drop_duplicates().reset_index(drop=True)
        val_topics['query'] = val_df['qid'].apply(lambda x: str(x))
        val_qrels = val_df[['qid', 'docno', 'label']].drop_duplicates().reset_index(drop=True)

        test_topics = test_df[['qid']].drop_duplicates().reset_index(drop=True)
        test_topics['query'] = test_df['qid'].apply(lambda x: str(x))
        test_qrels = test_df[['qid', 'docno', 'label']].drop_duplicates().reset_index(drop=True)

        # drop label and comment in original df
        train_df = train_df.drop(['label', 'comment:'], axis=1)
        val_df = val_df.drop(['label', 'comment:'], axis=1)
        test_df = test_df.drop(['label', 'comment:'], axis=1)

        all_data = pd.concat([train_df, val_df, test_df], ignore_index=True, axis=0)

        all_qrels = pd.concat([train_qrels, val_qrels, test_qrels], ignore_index=True, axis=0)

        QidLookupTransformer = pt.Transformer.from_df(all_data, uniform=True)
        pipeline = QidLookupTransformer

        # lambdamart
        lmart_x = xgb.sklearn.XGBRanker(objective='rank:ndcg',
                                        learning_rate=0.1,
                                        gamma=1.0,
                                        min_child_weight=0.1,
                                        max_depth=3,
                                        verbose=2,
                                        random_state=42
                                        )
        lmart_x_pipe = pipeline >> pt.ltr.apply_learned_model(lmart_x, form="ltr")

        # fast rank
        train_request = fastrank.TrainRequest.coordinate_ascent()
        params = train_request.params
        params.init_random = True
        params.normalize = True
        params.seed = 1234567
        fr_pipe = pipeline >> pt.ltr.apply_learned_model(train_request, form='fastrank')

        # fit model
        lmart_x_pipe.fit(train_topics, all_qrels, val_topics, all_qrels)
        fr_pipe.fit(train_topics, all_qrels)

        # extract feature importance
        lmart_x_feature = lmart_x_pipe[1].learner.feature_importances_
        fr_feature = np.array(fr_pipe[1].model.to_dict()['Linear']['weights'])

        if lmart_x_feature_importance is None:
            lmart_x_feature_importance = lmart_x_feature
        else:
            lmart_x_feature_importance = lmart_x_feature_importance + lmart_x_feature

        if fr_feature_importance is None:
            fr_feature_importance = fr_feature
        else:
            fr_feature_importance = fr_feature_importance + fr_feature

        # convert label to int (required by pt.Experiment())
        all_qrels['label'] = all_qrels['label'].astype('int')
        # Start experiment
        result = pt.Experiment(
            [lmart_x_pipe, fr_pipe],
            test_topics, all_qrels,
            eval_metrics=[ir_measures.RR(rel=1), ir_measures.nDCG@10, ir_measures.MAP(rel=1)],
            names=["LambdaMART", "FastRank"])

        lmart_x_ndcg += result['nDCG@10'][0]
        fr_ndcg += result['nDCG@10'][1]
        lmart_x_map += result['AP'][0]
        fr_map += result['AP'][1]

    # average
    lmart_x_mean_feature = lmart_x_feature_importance / 5
    fr_mean_feature = fr_feature_importance / 5

    lmart_x_top = np.argsort(lmart_x_mean_feature)[::-1][:10]
    fr_top = np.argsort(fr_mean_feature)[::-1][:10]

    lmart_x_mean_ndcg = lmart_x_ndcg / 5
    lmart_x_mean_map = lmart_x_map / 5
    fr_mean_ndcg = fr_ndcg / 5
    fr_mean_map = fr_map / 5

    print('LambdaMART: ')
    print(f'Mean nDCG: {lmart_x_mean_ndcg}')
    print(f'Mean MAP: {lmart_x_mean_map}')
    print(f'Top 10 important features: {lmart_x_top}')
    print(f'Mean Feature Importance: {lmart_x_mean_feature}')

    print('------------------------------------------------------------')
    print('FastRank: ')
    print(f'Mean nDCG: {fr_mean_ndcg}')
    print(f'Mean MAP: {fr_mean_map}')
    print(f'Top 10 important features: {fr_top}')
    print(f'Mean Feature Importance: {fr_mean_feature}')
