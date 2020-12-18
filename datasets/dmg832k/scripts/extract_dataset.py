#!/usr/bin/env python

import csv
import gzip
import sys

from rdflib import Graph
from rdflib.term import BNode, Literal, URIRef
from rdflib.util import guess_format


class IndexedMap():
    _map = None
    _map_rev = None
    _num_items = -1

    def __init__(self):
        self._map = dict()
        self._map_rev = dict()
        self._num_items = 0

    def add(self, item):
        if item in self._map.values():
            return (self._map_rev[item], False)

        index = self._num_items
        self._map[index] = item
        self._map_rev[item] = index

        self._num_items += 1

        return (index, True)

    def rev_map(self):
        return self._map_rev

    def __len__(self):
        return self._num_items

class Writer():
    _file = None
    _csv = None

    def __init__(self, path, header, gzipped=False):
        if gzipped:
            self._file = gzip.open(path, 'wt', encoding='utf-8')
        else:
            self._file = open(path, 'w', encoding='utf-8')

        self._csv = csv.writer(self._file, delimiter=',')
        self.write(header)

    def write(self, lst):
        self._csv.writerow(lst)

    def close(self):
        self._file.close()

def import_graphs(paths):
    graphs = dict()
    for i, g in enumerate(['context',
                           'training',
                           'testing',
                           'validation']):
        filename = paths[i]
        fileformat = guess_format(filename[:-3])
        with gzip.open(filename, 'rb') as gzf:
            graphs[g] = Graph()
            graphs[g].parse(gzf, format=fileformat)

    return graphs

def add_to_map(item, dictionary):
    if item in dictionary.values():
        return

def index_graph(graphs):
    resources = IndexedMap()
    properties = IndexedMap()
    nodetypes = IndexedMap()
    languages = IndexedMap()

    triples_writer = Writer('./triples.int.csv.gz', ["subject", "predicate",
                                                     "object"], gzipped=True)
    resource_writer = Writer('./nodes.int.csv', ["index", "nodetype",
                                                "label"])
    properties_writer = Writer('./relations.int.csv', ["index", "label"])
    nodetypes_writer = Writer('./nodetypes.int.csv', ["index", "label"])

    # add basic types
    unknown_idx, _ = nodetypes.add('unknown')
    nodetypes_writer.write([unknown_idx, 'unknown'])

    bnode_idx, _ = nodetypes.add('blank_node')
    nodetypes_writer.write([bnode_idx, 'blank_node'])

    iri_idx, _ = nodetypes.add('iri')
    nodetypes_writer.write([iri_idx, 'iri'])

    for g in graphs.values():
        for s, p, o in g.triples((None, None, None)):
            s_idx, s_new = resources.add(s)
            p_idx, p_new = properties.add(p)
            o_idx, o_new = resources.add(o)

            # write triple
            triples_writer.write([s_idx, p_idx, o_idx])

            # write property if unseen
            if p_new:
                properties_writer.write([p_idx, p])

            # write resources if unseen
            if s_new:
                if isinstance(s, URIRef):
                    s_type_idx = iri_idx
                elif isinstance(s, BNode):
                    s_type_idx = bnode_idx
                resource_writer.write([s_idx, s_type_idx, s])

            if o_new:
                o_type_idx = None
                if isinstance(o, URIRef):
                    o_type_idx = iri_idx
                elif isinstance(o, BNode):
                    o_type_idx = bnode_idx

                if isinstance(o, Literal):
                    o_type_idx, o_type_new = None, False

                    dtype = o.datatype
                    if dtype is not None:
                        o_type_idx, o_type_new = nodetypes.add(dtype)
                        if o_type_new:
                            nodetypes_writer.write([o_type_idx, dtype])

                    lang = o.language
                    if lang is not None:
                        o_type_idx, o_type_new = languages.add(lang)
                        if o_type_new:
                            nodetypes_writer.write([o_type_idx, lang])

                    o = '"{}"'.format(str(o))

                if o_type_idx is None:
                    o_type_idx = unknown_idx

                resource_writer.write([o_idx, o_type_idx, o])

    triples_writer.close()
    resource_writer.close()
    properties_writer.close()
    nodetypes_writer.close()

    return resources

def index_samples(graphs, resources):
    classes = IndexedMap()
    classes_writer = Writer('./classes.int.csv', ["index", "label"])
    samples_writer = Writer('./all.int.csv', ["index", "class"])

    for split in ['training', 'testing', 'validation']:
        writer = Writer('./'+split+'.int.csv', ["index", "class"])
        g = graphs[split]

        for s, _, o in g.triples((None, None, None)):
            class_idx, class_new = classes.add(o)
            if class_new:
                classes_writer.write([class_idx, o])

            sample_idx = resources.rev_map()[s]
            writer.write([sample_idx, class_idx])
            samples_writer.write([sample_idx, class_idx])

def main(graphs):
    resources = index_graph(graphs)
    index_samples(graphs, resources)

if __name__ == "__main__":
    assert len(sys.argv) == 5, "Expecting <context graph> <training triples>"\
                               " <testing triples> <validation triples>"

    graphs = import_graphs(sys.argv[1:])
    main(graphs)

