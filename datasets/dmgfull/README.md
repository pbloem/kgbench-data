# Dutch Monument Graph (DMG)

A multimodal knowledge graph about monuments in The Netherlands. This graph integrates open data from several Dutch
(semi-) government institutes and organizations (listed below), and includes (base64-encoded) images and geometries from
63,566 registered monuments as well as detailed contextual information.

A subset of this graph used as benchmark dataset for multimodal learning on heterogeneous knowledge graphs can be found in `../dmg832k/`.

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

`raw/dmgFull_stripped.hdt`
- DMG benchmark dataset in HDT format for multimodal learning on knowledge graphs; stripped of target samples

`raw/dmgFull_stripped.nt.gz`
- DMG benchmark dataset in N-Triple for multimodal learning on knowledge graphs; stripped of target samples

`raw/train_set.nt.gz`
- training target samples, stripped from dataset, stratified 

`raw/valid_set.nt.gz`
- validation target samples, stripped from dataset, stratified

`raw/test_set.nt.gz`
- test target samples, stripped from dataset, stratified

`raw/meta_set.nt.gz`
- meta (test) target samples, stripped from dataset, stratified

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
| Facts     | 1,882,441 |
| Relations | 63        |
| Entities  | 262,494   |
| Literals  | 1,289,150 |

| Modality  | Count     |
|-----------|-----------|
| Numerical | 64,224    |
| Temporal  | 23,944    |
| Textual   | 1,006,486 |
| Visual    | 58,846    |
| Spatial   | 121,217   |
| Boolean   | 8,838     |


