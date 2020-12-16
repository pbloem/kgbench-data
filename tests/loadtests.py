import unittest
import torch
from torch import nn

import sys

import kgbench as kg
from kgbench import tic, toc

class TestLoad(unittest.TestCase):

    stats = {
        'amfull' : {
                'num edges' : 5988321,
                'num relations' : 133,
                'num entities' : 1666764,
                'num labeled': 73423,
                'num classes': 5
        },
        'am1k': {
            'num edges': 5988321,
            'num relations': 133,
            'num entities': 1666764,
            'num labeled':  1000,
            'num classes': 11
        }
    }

    def test_load(self):

        for name in [ 'am1k', 'amfull']:
            print(f'checking {name}.')

            data = kg.load(name, final=True)
            # we load in final mode, since the stats above are for the complete data

            tic()

            assert data.num_relations == len(data.i2r) == len({p for _, p, _ in data.triples}) == data.triples.max(axis=0)[1] + 1, \
                f'dataset {name} {len(data.i2r)=}\t{len({p for _, p, _ in data.triples})=}\t{data.triples.max(axis=0)[1] + 1=} '
            e = set()
            for s, _, o in data.triples:
                e.add(s); e.add(o)

            assert data.num_entities == len(data.i2e) == len(e) == max(data.triples.max(axis=0)[0], data.triples.max(axis=0)[2]) + 1

            assert self.stats[name]['num edges'] == data.triples.shape[0]
            assert self.stats[name]['num relations'] == data.num_relations
            assert self.stats[name]['num entities'] == data.num_entities, \
                f'{self.stats[name]["num entities"]=} vs {data.num_entities=}'
            assert self.stats[name]['num labeled'] == data.training.shape[0] + data.withheld.shape[0], \
                f'{self.stats[name]["num labeled"]} {data.training.shape[0] + data.withheld.shape[0]=}'
            assert self.stats[name]['num classes'] == data.num_classes == len({c for c in data.training[:, 1]})

            print(f'checked {name} ({toc():.4}s).')

    def test_mm(self):
        """
        Testing the loading functions of multimodal data
        :return:
        """

        amplus = kg.load('amplus')

        ims = amplus.get_images()

        ims[6467].save('test1.png')
        ims[7468].save('test2.png')
        ims[9999].save('test3.png')

        # ims = amplus.get_image_tensor()
        # print(ims.shape)
        # print(ims.min(), ims.max(), ims.mean())

    def test_dtk(self):

        strings = ['uri', 'aaaa', 'bbbb', 'bb', 'z', 'none', 'blank_node']
        strings.sort(key=kg.datatype_key)

        for s in strings:
            print(s)



