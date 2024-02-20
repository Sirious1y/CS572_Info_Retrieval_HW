import os
import pyterrier as pt


# def bm25_finetune(index, topic, qrels):
#     bm25 = pt.BatchRetrieve(index, wmodel="BM25")
#
#     exp = pt.Experiment([bm25], topic, qrels, eval_metrics=["map"], verbose=True)

fb_terms = [1, 10, 20, 50, 100]
fb_docs = [1, 3, 5, 10, 20, 50, 100]
fb_lambda = [0.1, 0.3, 0.5, 0.9]

def rm3_finetune(index, topic, qrels):

    bm25 = pt.BatchRetrieve(index, wmodel="BM25")

    best_param = 0
    best_map = 0

    for terms in fb_terms:
        rm3_pipe = bm25 >> pt.rewrite.RM3(index, fb_terms=terms) >> bm25
        exp = pt.Experiment([rm3_pipe], topic, qrels, eval_metrics=['map'])

        print(f'term: {terms}, MAP: {float(exp["map"][0])}')

        if float(exp['map'][0]) > best_map:
            best_param = terms
            best_map = exp['map'][0]

    print(f'Best fb_terms: {best_param}, Best MAP: {best_map}')
    # best fb_terms is 10 with map=0.212881
    terms = best_param

    best_param = 0
    best_map = 0
    for docs in fb_docs:
        rm3_pipe = bm25 >> pt.rewrite.RM3(index, fb_terms=terms, fb_docs=docs) >> bm25
        exp = pt.Experiment([rm3_pipe], topic, qrels, eval_metrics=['map'])

        print(f'docs: {docs}, MAP: {float(exp["map"][0])}')

        if float(exp['map'][0]) > best_map:
            best_param = docs
            best_map = exp['map'][0]

    print(f'Best fb_docs: {best_param}, Best MAP: {best_map}')
    # best fb_docs is 3 with map=0.212881
    docs = best_param

    best_param = 0
    best_map = 0
    for lm in fb_lambda:
        rm3_pipe = bm25 >> pt.rewrite.RM3(index, fb_terms=terms, fb_docs=docs, fb_lambda=lm) >> bm25
        exp = pt.Experiment([rm3_pipe], topic, qrels, eval_metrics=['map'])

        print(f'lambda: {lm}, MAP: {float(exp["map"][0])}')

        if float(exp['map'][0]) > best_map:
            best_param = lm
            best_map = exp['map'][0]

    print(f'Best fb_lambda: {best_param}, Best MAP: {best_map}')
    # best fb_lambda is 0.3 with map=0.227261
    lm = best_param

    return terms, docs, lm


def rm3_result(index, topic, qrels, terms, docs, lm):
    bm25 = pt.BatchRetrieve(index, wmodel="BM25")
    rm3_pipe = bm25 >> pt.rewrite.RM3(index, fb_terms=terms, fb_docs=docs, fb_lambda=lm) >> bm25
    rm1_pipe = bm25 >> pt.rewrite.RM3(index, fb_terms=terms, fb_docs=docs, fb_lambda=0) >> bm25

    metrics = ['P_5', 'recall_5', 'P_10', 'recall_10', 'P_20', 'recall_20', 'P_100', 'recall_100', 'map', 'ndcg']

    return pt.Experiment([bm25, rm1_pipe, rm3_pipe], topic, qrels, eval_metrics=metrics, names=['bm25', 'rm1', 'rm3'], verbose=True)


if __name__ == "__main__":
    pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])

    index_path = "E:/Emory/2024/CS572/P1/text_index/"
    index_path = os.path.abspath(index_path)

    topic_file = "./test/topics.351-400"
    qrel_file = "./test/qrels.trec7.all"

    index = pt.IndexFactory.of(index_path)
    topic = pt.io.read_topics(topic_file)
    qrels = pt.io.read_qrels(qrel_file)

    best_terms, best_docs, best_lambda = rm3_finetune(index, topic, qrels)

    topic_file = "./test/topics.401-450"
    qrel_file = "./test/qrels.trec8.all"
    topic = pt.io.read_topics(topic_file)
    qrels = pt.io.read_qrels(qrel_file)

    exp = rm3_result(index, topic, qrels, best_terms, best_docs, best_lambda)

    with open('results_rm3.txt', 'w') as f:
        result = exp.to_string(header=True, index=False)
        f.write(result)
        f.write('\n')
        f.write(f'fb_terms search space: {fb_terms}\n')
        f.write(f'best fb_terms: {best_terms}\n')
        f.write(f'fb_docs search space: {fb_docs}\n')
        f.write(f'best fb_docs: {best_docs}\n')
        f.write(f'fb_lambda search space: {fb_lambda}\n')
        f.write(f'best fb_lambda: {best_lambda}')
