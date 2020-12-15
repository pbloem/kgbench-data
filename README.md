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

## amfull

Like the Legacy AM dataset (in the repository as am1k), this features the metadata for the collection of the Amsterdam Museum. The classification task is to 


# Datatypes

To describe the various modalities in the data, existing datatype specifications aren't always sufficient. For our 
purposes, we introduce the new datatype entities for media in byte representation, string in a string-encoded form in a 
literal. They have the following format

```http://krrvu.github.io/kgbench/dt#[media]-[string-encoding][-media-encoding]```

Media types can be one of: `image`, `audio`, `video`. `string-encoding` is `b64string` for a base64 encoding. Other encodings are currently not specified.

The media-encoding may be ommitted if it can be inferred from the byte-level representation of the file. If it is present 
it serves as a _hint_ towards the way the byte-level representation should be decoded, in the same way that file extensions do

Some examples:
```
# for an image without specified codec
http://krrvu.github.io/kgbench/dt#image-b64string

# for a jpg-encoded image
http://krrvu.github.io/kgbench/dt#image-b64string-jpg

# for a wav-encoded audio file
http://krrvu.github.io/kgbench/dt#audio-b64string-wav

# for a video file we specify either no internal encoding or the container type
http://krrvu.github.io/kgbench/dt#video-b64string
http://krrvu.github.io/kgbench/dt#video-b64string-mkv
```

## Legacy

The following legacy datasets are added as a record, and to check the correct implementations of our baselines. In general they are less suitable for benchmarking than the datasets described above. 