#!/usr/bin/env python

import sys

# https://github.com/Callidon/pyHDT
import hdt
import numpy as np
from tqdm import tqdm

def generate_stats(doc):
    n_edges = len(doc)
    n_vertices = 0

    # create integer mapping
    vertices = set()
    triples, c = doc.search_triples('', '', '')
    for s, p, o in tqdm(triples, total=c):
        vertices.add(s)
        vertices.add(o)

    n_vertices = len(vertices)
    vertex_idx_map = {vertex:i for i, vertex in enumerate(vertices)}

    # compute degrees
    degree_array = np.zeros((n_vertices, 3), dtype=int)
    triples, c = doc.search_triples('', '', '')
    for s, p, o in tqdm(triples, total=c):
        s_idx = vertex_idx_map[s]
        o_idx = vertex_idx_map[o]

        degree_array[s_idx, 0] += 1  # outdegree
        degree_array[o_idx, 1] += 1  # indegree

    # overal degree
    degree_array[:,2] = degree_array[:, 0] + degree_array[:, 1]

    sys.stdout.write('- degree: min %d / max %d / avg %f\n' % (np.min(degree_array[:,2]),
                                                             np.max(degree_array[:,2]),
                                                             np.mean(degree_array[:,2])))

    sys.stdout.write('- in degree: min %d / max %d / avg %f\n' % (np.min(degree_array[:,1]),
                                                             np.max(degree_array[:,1]),
                                                             np.mean(degree_array[:,1])))

    sys.stdout.write('- out degree: min %d / max %d / avg %f\n' % (np.min(degree_array[:,0]),
                                                             np.max(degree_array[:,0]),
                                                             np.mean(degree_array[:,0])))

    # compute density
    sys.stdout.write('- density: %f\n' % density(n_vertices, n_edges))

def density(num_vertices, num_edges):
    return num_edges / (num_vertices * (num_vertices - 1))

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 1:
        print("USAGE: ./graphstats.py <graph_stripped.hdt>")

    hdtfile = args[0]
    doc = hdt.HDTDocument(hdtfile)

    generate_stats(doc)

