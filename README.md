# kgbench

A set of benchmark repositories for node classification on knowledge graphs. Paper:
[kgbench: A Collection of Datasets for Multimodal and Relational Learning on Heterogeneous Knowledge](https://openreview.net/forum?id=yeK_9wxRDbA) 

We offer a set of node classification benchmark tasks on relational data, with the following aims:

* A large number of labeled nodes. In particular a large _test set_, so that accuracy can be precisely estimated. In general, we aim for 10K instances in the test set set. We will prioritize a large test set over a large training set.
* A small number of well-balanced classes, with many instances per class. If possible, we manually merge the given categories to higher-level classes in order to create better balanced classes (where this makes sense semantically). 
* A medium-sized knowledge graph, so that ambitious models can be trained quickly. For some graphs, a full-batch 2 layer RGCN model fits in the memory of a 12 Gb GPU.
* A diverse selection of domains: movies, cultural heritage, museum collection data and academic publication data.
* Diverse multimodal literals that can (optionally) be used to improve performance. Including natural language, dates, images and spatial geometries.  

## Installation

Download or clone the repository https://github.com/pbloem/kgbench-loader. In the root directory (where `setup.py` is located), run 
```
pip install . 
```

_Please do not use the `kgbench-data` repository, only use the `kgbench-loader`. The former should only be used to study how the data was created._

## Loading data in python

The following snippet loads the `amplus` dataset

```python
import kgbench as kg

data = kg.load('amplus') # Load with numpy arrays, and train/validation split

data = kg.load('amplus', torch=True) # Load with pytorch arrays

data = kg.load('amplus', final=True) # Load with numpy arrays and train/test split
``` 

The `data` object contains all relevant information contained in the dataset, preprocessed for direct use in pytorch or in any numpy-based machine learning framework.

The following are the most important attributes of the `data` object:

 * `data.num_entities` Total number of distinct nodes (literals or entities) in the graph.
 * `data.num_relations` Total number of distinct relation types in the graph.
 * `data.num_classes` Total number of classes in the classification task.
 * `data.triples` The edges of the knowledge graph (the triples), represented by their integer indices. An `(m, 3)` numpy or pytorch array.
 * `data.training` Training labels: a matrix with node indices in column 0 and class indices in column 1.
 * `data.withheld` Validation/testing data: a matrix with entity indices in column 0 and class indices in column 1. In non-final mode this is the validation data. In final mode this is the testing data.

These are all the attributes required to implement a classifier for the **relational setting**. That is, the setting where literals are treated as atomic nodes. In the **multimodal setting**, where the content of literals is also taken into account, the following attributes and methods can also be used.

 * `data.i2r, data.r2i` A list and dictionary respectively, mapping relation indices to string representations of the relation (or predicate). 
 * `data.i2n` A mapping from an integer index to a node representation: a pair indicating the annotation and the label (in that order). Annotations can be 'iri', 'blank_node', 'none' (untagged literal) a language tag, or a datatype IRI.
 * `data.n2i` The inverse mapping of `data.i2n`. Note that the keys are pairs.
 * `data.datatype_g2l(dtype)` Returns a list mapping a global index of an entity (the indexing over all nodes) to its _local index_ the indexing over all nodes of the given datatype.
 * `data.datatype_l2g(dtype)` Maps local to global indices.
 * `data.get_images()` Returns the images in the dataset (in order of local index) as a list of PIL image objects. Utility function are provided to process and batch these (see the mrgcn experiment for an example).

The `scripts` directory contains the scripts needed to convert any RDF knowledge graph to the format listed above, allowing it to be imported using the kgbench dataloader.

## Experiments

Three example baselines are implemented in the directory `experiments`. These should give a fairly complete idea of the way the library can be used. See the paper for model details. 

## Loading data in other languages

If you aren't working in python, you'll have to load the data yourself. This can be done with any standard CSV loader.

Each datasets is laid out in the following files:
 * `triples.int.csv.gz` A gzipped CSV file of the triples represented as integers. Each node in the graph and each relation is assigned an integer. Since, in many case, you won't need more than the node and relation identites, this file is all you need to load the graph. Note that this CSV file has no headers.
 * `training.int.csv`, `validation.int.csv` , `testing.int.csv` The node labels. The first column is the node index, the second is the class. Note that these CSV files have headers.
 * `meta-testing.int.csv` Metatesting split. You probably don't need to touch this.
 * `nodes.int.csv` String representations and annotations for each node index.
 * `nodetypes.int.csv` Indexed node and annotation types.
 * `relations.int.csv` String representations for each relation.
 
## RDF Data

For each dataset, the original RDF is available, together with the scripts that extract the CSV form in a directory named `raw`. Since the sources of the data differ per dataset, what is contained in this directory differs per dataset.  

## IMPORTANT Notes for use

Make sure to follow these instructions to run a correct experiment that is comparable to other experiments on these datasets:
 * Do not use the test data more than once per project/report/model. Choose your hyperparameters based on the validation data, and report performance on the test data. 
 * When running the final experiment on the test data, **do not combine the train and validation data** for a larger training set. This is common practice in many settings, but in this case, since the validation data is so large, it offers too great an advantage. For the final run, training should be done only on the training data. 

# Datasets

The following benchmark datasets are available. See the paper for more extensive descriptions. 

 * `amplus` Extended version of the AM data
 * `dblp` Subset of the DBLP database, enriched with author information from Wikidata
 * `dmgfull` Monuments in the Netherlands.
 * `dmg777k` Subset of `dmgfull`
 * `mdgenre` Movie data extracted from Wikidata
 
The following datasets are available for unit testing:

 * `micro` Minute dataset of 5 nodes. Useful for manually analyzing algorithm behavior.
 * `aifb` Small, real-world dataset. The test set is too small to make this a very valuable benchmark, but good for quick sanity checks.
 * `mdgender` Large, balanced multimodal dataset with a guaranteed relation between the images and the target class, which a convolutional network can learn. This task is too easy to make for a good evaluation, but it may be helpful in troubleshooting multimodal models. See the broader impact statement in the paper for a discussion on the problems surrounding gender classification.  

## Datatypes

KGBench datasets contain byte-to-string encoded literals. These string literals encode byte-level data, potentially containing images, video or audio (although only images are currently used in the datasets).

We define the following datatypes:
 
 * [http://kgbench.info/dt#base64Image](http://kgbench.info/dt.ttl) An image encoded as a base64 string. 
 * [http://kgbench.info/dt#base64Audio](http://kgbench.info/dt.ttl) An audio sequence encoded as a base64 string.
 * [http://kgbench.info/dt#base64Video](http://kgbench.info/dt.ttl) A video encoded as a base64 string.

In most cases this information is sufficient to correctly decode the byte-level information. To provide a fully unambiguous definition of how a literal should be decoded, it is necessary also to specify its MIME-type. This can be done by adding extra statements to the graph, but this is outside the scope of the `kgbench` project. 

In our datasets, every media type uses a uniform choice of codec (that is, all images are either JPEG or PNG, but these are not mixed within one dataset). This choice is specified in the dataset metadata.
