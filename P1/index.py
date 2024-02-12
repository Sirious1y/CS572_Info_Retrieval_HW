import os
import pyterrier as pt

# os.environ["JAVA_HOME"] = "/Program Files/Common Files/Java/jdk-21/"
pt.init()

files = pt.io.find_files("./text/ft/")

index_folder = "C:/MyThing/Emory/2024/CS572/P1/ft_index/"
index_path = os.path.abspath(index_folder)

print(files)

indexer = pt.TRECCollectionIndexer(index_path, verbose=True, blocks=False)
indexref = indexer.index(files)