# Dutch Monument Graph (DMG)

A multimodal knowledge graph about monuments in The Netherlands. This graph integrates open data from several Dutch (semi-)
government institutes and organizations (listed below), and includes (base64-encoded) images and geometries from 63,566
registered monuments as well as detailed contextual information.

A subset of this graph used as benchmark dataset for multimodal learning on heterogeneous knowledge graphs can be found
in `/dmg832k/`.

## Data Providers

* [Rijksdienst voor het Cultureel Erfgoed](https://www.cultureelerfgoed.nl) ([Beeldbank](https://beeldbank.cultureelerfgoed.nl))
* [Centraal Bureau voor de Statistiek ](https://www.cbs.nl)
* [Kadaster](https://www.kadaster.nl)
* [Geonames](https://www.geonames.org)

## Statistics

| Stats     | Count     |
|-----------|-----------|
| Facts     | 2,119,187 |
| Relations | 64        |
| Entities  | 262,495   |
| Literals  | 1,462,331 |

| Modality  | Count     |
|-----------|-----------|
| Numerical | 64,224    |
| Temporal  | 23,944    |
| Textual   | 1,179,667 |
| Visual    | 58,846    |
| Spatial   | 121,217   |
| Boolean   | 8,838     |


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
