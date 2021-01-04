# Scripts

The scripts listed here can be used to convert any N-Triple or HDT graph to the KGBench format.

## USAGE

The following examples assume the AIFB dataset.

1. Strip the targets from the original graph (outputs to aifb_stripped.hdt) and store these in `samples.nt.gz`:
 
`./strip_targets.sh aifb.nt http://swrc.ontoware.org/ontology#employs`

2. Generate classification splits (stratified; 50 test, 20 valid, remaining train) using the extracted targets, and
   store these as <split>.nt.gz:
   
`python mksplits_classification.py samples.nt.gz 50 20`

or with a separate meta set of 30 samples:

`python mksplits_classification.py samples.nt.gz 50 20 30`

3. Convert aifb_stripped.hdt and its 3 or 4 splits to the KGBench format:

`python hdt2csv.py aifb_stripped.hdt train_set.nt.gz test_set.nt.gz valid_set.nt.gz meta_set.nt.gz`

4. (Optional) Generate link-prediction splits (stratified; 500 test, 200 valid, remaining train) using
   `triples.int.csv.gz`:
   
`python mksplits_linkprediction.py triples.int.csv.gz 500 200`

or with a separate meta set of 300 samples:

`python mksplits_linkprediction.py triples.int.csv.gz 500 200 300`


