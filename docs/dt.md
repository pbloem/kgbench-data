# KGBench datatype

KGBench datasets contain byte-to-string encoded literals. These string literals encode byte-level data, potentially containing images, video or audio (although only images are currently used in the datasets).

We define the following datatypes:
 
 * [http://kgbench.info/dt#base64Image](http://kgbench.info/dt.ttl) An image encoded as a base64 string. 
 * [http://kgbench.info/dt#base64Audio](http://kgbench.info/dt.ttl) An audio sequence encoded as a base64 string.
 * [http://kgbench.info/dt#base64Video](http://kgbench.info/dt.ttl) A video encoded as a base64String.

In most cases this information is sufficient to correctly decode the byte-level information. To provide a fully unambiguous definition of how a literal should be decoded, it is necessary also to specify its MIME-type. This can be done by adding extra statements to the graph, but this is outside the scope of the `kgbench` project. 

In our datasets, every media type uses a uniform choice of codec (that is, all images are either JPEG or PNG, but these are not mixed within one dataset). This choice is specified in the dataset metadata.
