import os
import pyterrier as pt


def bm25_experiment(index_path, topic_file, qrel_file):
    index = pt.IndexFactory.of(index_path)
    bm25 = pt.BatchRetrieve(index, wmodel="BM25")


    topic = pt.io.read_topics(topic_file)
    qrels = pt.io.read_qrels(qrel_file)

    metrics = ['P_5', 'recall_5', 'P_10', 'recall_10', 'P_20', 'recall_20', 'P_100', 'recall_100', 'map', 'ndcg']

    return pt.Experiment([bm25], topic, qrels, eval_metrics=metrics, verbose=True)


if __name__ == "__main__":
    pt.init()

    index_path = "E:/Emory/2024/CS572/P1/text_index/"
    index_path = os.path.abspath(index_path)

    topic_file = "./test/topics.401-450"
    qrel_file = "./test/qrels.trec8.all"

    exp = bm25_experiment(index_path, topic_file, qrel_file)

    with open('results_bm25.txt', 'w') as f:
        result = exp.to_string(header=True, index=False)
        f.write(result)
