# kgbench

A set of benchmark repositories for node classification and link prediction on knowledge graphs. 

The aim is to create a set of machine-learning benchmark datasets with the following characteristics:

* A large number of labeled nodes. In particular a large test set, so that accuracy can be precisely estimated. In general, we aim for 10K instances in the test set set. We will prioritize a large test set over a large training set.
* A small number of well-balanced classes, with many instances per class. If possible, we manually merge the given categories to higher-level classes in order to create better balanced classes (where this makes sense semantically). 
* A medium-sized knowledge graph, so that ambitious models can be trained quickly. IOn general we aim for graphs where a full-batch model of the two hop neighborhood fits in the memory of a 12 Gb GPU.
* A diverse selection of domains and modalities.  

## Preprocessing and organization

Each dataset is stored under the direction `datasets`, each in its own directory.  

# Datasets

## amfull

Like the Legacy AM dataset (in the repository as am1k), this features the metadata for the collection of the Amsterdam Museum. The classification task is to 

## dmgFull

The Dutch Monument Graph (DMG) is a highly multimodal dataset which features information about monumental buildings in
The Netherlands. Unique about this dataset is the strong presence of _spatial information_, such as coordinates,
building foot prints (polygons), and various hierarchical relations (cities, municipalities, etc). If used for
classification, the task is to predict the type of monuments, such as `building`, `castle`, or `wind mill`.

### dmg832k

A subset of the DMG which only contains information about the top-5 monument classes. This version is suitable for GPU
use.


# Datatypes

A common way to represent large binary objects (BLOBs) in datasets is to represent these as byte-encoded strings.
In knowledge graphs, this can be done by encoding these string as literals, and by annotating them using e.g. XSD's
`base64Binary` datatype. However, existing datatype specifications aren't always sufficient, most particular when we
want to encode _different forms_ of binary data such as images, videos, and audio sequences. 

For our purposes, we introduce several new datatypes to annotate binary-encoded media with. Here, we limit ourselves to
three datatypes that are currently popular topics for machine learning, and for which no convention as of yet
exists in the Semantic Web community. 

We define the following datatypes:
 
 * [http://kgbench.info/dt#base64Image](http://kgbench.info/dt.ttl) An image encoded as a base64 string. 
 * [http://kgbench.info/dt#base64Audio](http://kgbench.info/dt.ttl) An audio sequence encoded as a base64 string.
 * [http://kgbench.info/dt#base64Video](http://kgbench.info/dt.ttl) A video encoded as a base64String.

In most cases this information is sufficient to correctly decode the byte-level information. To provide a fully unambiguous definition of how a literal should be decoded, it is necessary also to specify its MIME-type. This can be done by adding extra statements to the graph, but this is outside the scope of the `kgbench` project. 

In our datasets, every media type uses a uniform choice of codec (that is, all images are either JPEG or PNG, but these are not mixed within one dataset). This choice is specified in the dataset metadata.


## Legacy

The following legacy datasets are added as a record, and to check the correct implementations of our baselines. In general they are less suitable for benchmarking than the datasets described above. 
