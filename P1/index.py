import os
import pyterrier as pt

pt.init()

text_files = ['ft', 'fbis', 'latimes']

for file_name in text_files:
    files = pt.io.find_files("./text/" + file_name + '/')

    index_path = "E:/Emory/2024/CS572/P1/" + file_name + "_index/"
    index_path = os.path.abspath(index_path)

    # dataset = pt.get_dataset("trec-robust-2004")

    indexer = pt.TRECCollectionIndexer(index_path, verbose=True, blocks=False)
    indexref = indexer.index(files)
