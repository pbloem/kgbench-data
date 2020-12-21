# Dutch Monument Graph (DMG)

This dataset is derived from the [DMG graph](https://gitlab.com/wxwilcke/dmg), and includes detailed information about 8,399 monuments in The
Netherlands on top of images and geometries.

`nodes.int.csv`
- node to integer mapping

`relations.int.csv`
- relation to integer mapping

`nodetypes.int.csv`
- node type to integer mapping

`triples.int.csv.gz`
- triples using integer mapping

`all.int.csv`
- samples to class using integer mapping

`train.int.csv`
- train samples to class using integer mapping

`test.int.csv`
- test samples to class using integer mapping

`valid.int.csv`
- valid samples to class using integer mapping

`linkprediction-splits.int.csv`
- triples to splits using integer mapping

`raw/dmg832k_stripped.hdt`
- DMG benchmark dataset in HDT format for multimodal learning on knowledge graphs; stripped of target samples

`raw/dmg832k_stripped.nt.gz`
- DMG benchmark dataset in N-Triple for multimodal learning on knowledge graphs; stripped of target samples

`raw/dmg832k_train_set.nt.gz`
- training target samples, stripped from dataset, stratified 

`raw/dmg832k_valid_set.nt.gz`
- validation target samples, stripped from dataset, stratified

`raw/dmg832k_test_set.nt.gz`
- test target samples, stripped from dataset, stratified

`scripts/mksplits.py`
- generate train, test, and validation splits

`scripts/to_ints.py`
- convert raw HDT graph to CSVs

## Data Providers

* [Rijksdienst voor het Cultureel Erfgoed](https://www.cultureelerfgoed.nl) ([Beeldbank](https://beeldbank.cultureelerfgoed.nl))
* [Centraal Bureau voor de Statistiek ](https://www.cbs.nl)
* [Kadaster](https://www.kadaster.nl)
* [Geonames](https://www.geonames.org)

## Statistics

| Stats     | Count     |
|-----------|-----------|
| Facts     | 777,124   |
| Relations | 60        |
| Entities  | 148,127   |
| Literals  | 488,745   |

| Modality  | Count   |
|-----------|---------|
| Numerical | 8,906   |
| Temporal  | 1,800   |
| Textual   | 398,938 |
| Visual    | 46,108  |
| Spatial   | 20,866  |
| Boolean   | 8,299   |

## Note

This variant of DMG will be renamed DMG777k (due to different counting method) in the period between acceptance and publication of the corresponding paper.
