import fire, sys, tqdm

import kgbench as kg

"""
Load a dataset and print statistics.
"""

def go(name='amplus'):
    kg.load(name)

    print('arguments: ', ' '.join([f'{k}={v}' for k, v in locals().items()]))

    data = kg.load(name, torch=False, prune_dist=None, final=False)

    print(f'{data.triples.shape[0]} triples')
    print(f'{data.num_relations} relations')
    print(f'{data.num_entities} nodes')
    print()

    literals = 0
    for datatype in data.datatypes():
        print(f'    {datatype}, {len(data.get_strings(datatype))} ')
        if datatype not in ['iri', 'blank_node']:
            literals += len(data.get_strings(datatype))

    print(f'{len(data.get_strings("iri")) + len(data.get_strings("blank_node"))} entities.')
    print(f'{literals} literals.')
    print()
    print(f'{data.num_classes} classes ')
    print(f'{data.training.shape[0]} training instances.')
    print(f'{data.withheld.shape[0]} validation instances.')

    data = kg.load(name, torch=False, prune_dist=None, final=False)
    print(f'{data.withheld.shape[0]} test instances.')

if __name__ == '__main__':

    fire.Fire(go)