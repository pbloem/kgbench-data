# kgbench

A set of benchmark repositories for node classification on knowledge graphs (work in progress). 

The aim s to create a set of node classification benchmark datasets with the following characteristics:

* A large number of labeled nodes. In particular a large test set, so that accuracy can be precisely estimated. In general, we aim for 10K instances in the test set set. We will prioritize a large test set over a large training set.
* A small number of well-balanced classes, with many instances per class. If possible, we manually merge the given categories to higher-level classes in order to create better balanced classes (where this makes sense semantically). 
* A medium-sized knowledge graph, so that ambitious models can be trained quickly. IOn general we aim for graphs where a full-batch model of the two hop neighborhood fits in the memory of a 12 Gb GPU.
* A diverse selection of domains.  

 

## Preprocessing and organization

Each dataset is stored under the direction `datasets`, each in its own directory. The  

# Datasets

## Novel

### amfull

Like the Legacy AM dataset (in the repository as am1k), this features the metadata for the collection of the Amsterdam Museum. The classification task is to 


## Legacy

The following legacy datasets are added as a record, and to check the correct implementations of our baselines. In general they are less suitable for benchmarking than the datasets described above. 