import ir_measures
import pyterrier as pt
from pyterrier.measures import *
import pyltr
import pandas as pd
import xgboost as xgb

pt.init()
dataset = pt.get_dataset("vaswani")
index = dataset.get_index()
# vaswani dataset provides an index, topics and qrels
#bm25 = pt.BatchRetrieve.from_dataset(dataset, wmodel="BM25")
#tf = pt.BatchRetrieve.from_dataset(dataset, wmodel="TF_IDF")
#pl2 = pt.BatchRetrieve.from_dataset(dataset, wmodel="PL2")
#dph = pt.BatchRetrieve.from_dataset(dataset, wmodel="DPH")
bm25 = pt.BatchRetrieve.from_dataset(dataset, "terrier_stemmed", wmodel="BM25")
tf_idf = pt.BatchRetrieve.from_dataset(dataset, "terrier_stemmed", wmodel="TF_IDF")
#sdm = pt.rewrite.SequentialDependence()
pl2 = pt.BatchRetrieve.from_dataset(dataset, "terrier_stemmed", wmodel="PL2")
bm25_qe = pt.BatchRetrieve.from_dataset(dataset, wmodel="BM25", controls={"qe":"on", "qemodel" : "Bo1"})
#dph = pt.BatchRetrieve.from_dataset(dataset, "terrier_stemmed", wmodel="DPH")

#index = dataset.get_index(variant="terrier_stemmed")


result = pt.Experiment(
    [bm25, tf_idf, pl2, bm25_qe],
    dataset.get_topics(),
    dataset.get_qrels(),
    eval_metrics=[ir_measures.RR(rel=1), ir_measures.nDCG@10, ir_measures.MAP(rel=1)]
)

print(result)

#this extracts 3 features [bm25, tf, pl2]
pipeline1 = bm25 >>  (bm25 ** bm25_qe ** tf_idf ** pl2 )
df1 = pipeline1.search("test")
#print(df1.head())


#models: from http://terrier.org/docs/current/javadoc/org/terrier/matching/models/package-summary.html
#this version is broken - feature values are wrong
#pipeline2 = pt.FeaturesBatchRetrieve(index, wmodel="BM25",
#                                     features=["WMODEL:BM25", "WMODEL:TF_IDF", "WMODEL:PL2", "WMODEL:DPH"])

#df2 = pipeline2.search("test")
#print(df2.head())

#create dataframe with columns: "query, qid" columns, query can be empty
all_topics = dataset.get_topics()
#create dataframe with columns: qid, docno, label
all_qrels = dataset.get_qrels()

#get features save in dataframe
docs_with_features = pipeline1.transform(all_topics)
print(docs_with_features.head())
print(all_topics)
print(all_qrels)

#following the example from here to create transformer to lookup features given qid:
# https://pyterrier.readthedocs.io/en/latest/transformer.html#mocking-transformers-from-dataframes
QidLookupTransformer = pt.Transformer.from_df(docs_with_features, uniform=True)

#pipeline to run
pipeline=QidLookupTransformer

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=3)

rf_pipe = pipeline >> pt.ltr.apply_learned_model(rf)

from sklearn.model_selection import train_test_split
train_topics, test_topics = train_test_split(all_topics, test_size=0.6, random_state=42)
#cheating version for sanity check: train/test on same topicsß
#train_topics = all_topics
#test_topics = all_topics


import fastrank
train_request = fastrank.TrainRequest.coordinate_ascent()
params = train_request.params
params.init_random = True
params.normalize = True
params.seed = 1234567
fr_pipe = pipeline >> pt.ltr.apply_learned_model(train_request, form='fastrank')

# this configures XGBoost as LambdaMART
lmart_x = xgb.sklearn.XGBRanker(objective='rank:ndcg',
      learning_rate=0.1,
      gamma=1.0,
      min_child_weight=0.1,
      max_depth=3,
      verbose=2,
      random_state=42)

lmart_x_pipe = pipeline >> pt.ltr.apply_learned_model(lmart_x, form="ltr")

print ("fitting LambdaMART")
lmart_x_pipe.fit(train_topics, all_qrels, test_topics, all_qrels)


print("fitting FastRank")
fr_pipe.fit(train_topics, all_qrels)

print("fitting Random Forest")
rf_pipe.fit(train_topics, all_qrels)


result = pt.Experiment(
    [bm25, rf_pipe, fr_pipe, lmart_x_pipe],
    test_topics, all_qrels,
    eval_metrics=[ir_measures.RR(rel=1), ir_measures.nDCG@10, ir_measures.MAP(rel=1)],
    names=["BM25 Baseline", "RandomForest", "FastRank", "LambdaMART"])

print(result)

quit()




