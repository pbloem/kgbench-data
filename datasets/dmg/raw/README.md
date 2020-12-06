# Dutch Monument Graph (DMG)

This dataset is derived from the [DMG315k graph](https://gitlab.com/wxwilcke/dmg/dmg832k), and includes detailed information about 8,690 monuments in The
Netherlands on top of images and geometries.

`dmg_context.nt.gz`
- DMG benchmark dataset for multimodal learning on knowledge graphs; stripped of target samples

`dmg_train_set.nt.gz`
- training target samples, stripped from dataset, stratified 

`dmg_valid_set.nt.gz`
- validation target samples, stripped from dataset, stratified

`dmg_test_set.nt.gz`
- test target samples, stripped from dataset, stratified

## Data Providers

* [Rijksdienst voor het Cultureel Erfgoed](https://www.cultureelerfgoed.nl) ([Beeldbank](https://beeldbank.cultureelerfgoed.nl))
* [Centraal Bureau voor de Statistiek ](https://www.cbs.nl)
* [Kadaster](https://www.kadaster.nl)
* [Geonames](https://www.geonames.org)

## Statistics

| Stats     | Count     |
|-----------|-----------|
| Facts     | 832,614   |
| Relations | 62        |
| Entities  | 148,132   |
| Literals  | 535,839   |

| Modality  | Count   |
|-----------|---------|
| Numerical | 8,906   |
| Temporal  | 1,800   |
| Textual   | 436,032 |
| Visual    | 46,108  |
| Spatial   | 20,866  |
| Boolean   | 8,299   |


## Citation

Please use the following reference if you use this dataset in your research:

```
@misc{wilcke2020dmg,
  title={Dutch Monument Graph: A Multimodal Knowledge Graph About Monuments In The Netherlands},
  author={Wilcke, WX and Bloem, P and de Boer, V and van 't Veer, RH},
  howpublished = {\url{https://gitlab.com/wxwilcke/dmg}},
  year={2020}
}
```
