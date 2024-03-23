import pandas as pd
import numpy as np
import os
import pyterrier as pt
from pyterrier.measures import *
import pyltr
import fastrank


pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])
pd.set_option('display.max_columns', None)

index_path = "E:/Emory/2024/CS572/P1/text_index/"
index_path = os.path.abspath(index_path)

train_topic_file = "./test/topics.351-400"
train_qrel_file = "./test/qrels.trec7.all"

index = pt.IndexFactory.of(index_path)
train_topic = pt.io.read_topics(train_topic_file)
train_qrels = pt.io.read_qrels(train_qrel_file)

test_topic_file = "./test/topics.401-450"
test_qrel_file = "./test/qrels.trec8.all"
test_topic = pt.io.read_topics(test_topic_file)
test_qrels = pt.io.read_qrels(test_qrel_file)

all_qrels = pd.concat([train_qrels, test_qrels], ignore_index=True, axis=0)
all_topics = pd.concat([train_topic, test_topic], ignore_index=True, axis=0)

#Scores features
bm25 = pt.BatchRetrieve(index, wmodel="BM25")
pl2 = pt.BatchRetrieve(index, wmodel="PL2")
rm3 = bm25 >> pt.rewrite.RM3(index) >> bm25

# score features experiment
score_feature_pipeline = bm25 >> (bm25 ** pl2 ** rm3)

score_feature_result = score_feature_pipeline.transform(all_topics)

QidLookupTransformer = pt.Transformer.from_df(score_feature_result, uniform=True)
pipeline = QidLookupTransformer

train_request = fastrank.TrainRequest.coordinate_ascent()
params = train_request.params
params.init_random = True
params.normalize = True
params.seed = 1234567
fr_pipe = pipeline >> pt.ltr.apply_learned_model(train_request, form='fastrank')

fr_pipe.fit(train_topic, all_qrels)
score_fr_feature = np.array(fr_pipe[1].model.to_dict()['Linear']['weights'])

result = pt.Experiment(
    [bm25, fr_pipe],
    test_topic, all_qrels,
    eval_metrics=[ir_measures.RR(rel=1), ir_measures.nDCG @ 10, ir_measures.MAP(rel=1)],
    names=["BM25 Baseline", "FastRank"])

score_fr_ndcg = result['nDCG@10'][1]
score_fr_map = result['AP'][1]


# Extended features
tf = pt.BatchRetrieve(index, wmodel="Tf")
tf_idf = pt.BatchRetrieve(index, wmodel="TF_IDF")
dlh = pt.BatchRetrieve(index, wmodel="DLH")
def rank(row):
    return np.array([row['rank']])

def query_len(row):
    return np.array([len(row['query'])])

rank_feature = pt.apply.doc_features(rank)
query_len_feature = pt.apply.doc_features(query_len)

extend_feature_pipeline = bm25 >> (bm25 ** pl2 ** rm3 ** tf ** tf_idf ** dlh ** rank_feature ** query_len_feature)

extend_feature_result = extend_feature_pipeline.transform(all_topics)

QidLookupTransformer = pt.Transformer.from_df(extend_feature_result, uniform=True)
pipeline = QidLookupTransformer

train_request = fastrank.TrainRequest.coordinate_ascent()
params = train_request.params
params.init_random = True
params.normalize = True
params.seed = 1234567
fr_pipe = pipeline >> pt.ltr.apply_learned_model(train_request, form='fastrank')

fr_pipe.fit(train_topic, all_qrels)
extend_fr_feature = np.array(fr_pipe[1].model.to_dict()['Linear']['weights'])

result = pt.Experiment(
    [bm25, fr_pipe],
    test_topic, all_qrels,
    eval_metrics=[ir_measures.RR(rel=1), ir_measures.nDCG @ 10, ir_measures.MAP(rel=1)],
    names=["BM25 Baseline", "FastRank"])

extend_fr_ndcg = result['nDCG@10'][1]
extend_fr_map = result['AP'][1]

print('Score Features [bm25, PL2, rm3]: ')
print(f'nDCG: {score_fr_ndcg}')
print(f'MAP: {score_fr_map}')
print(f'Feature Importantce: {score_fr_feature}')
print('-------------------------------------------------------------------')
print('Extend Features: [bm25, PL2, rm3, tf, tf_idf, dlh, rank, query length]')
print(f'nDCG: {extend_fr_ndcg}')
print(f'MAP: {extend_fr_map}')
print(f'Feature Importantce: {extend_fr_feature}')


