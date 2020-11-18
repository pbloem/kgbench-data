import os
import torch

import rdflib as rdf

import time

tics = []

def tic():
    tics.append(time.time())

def toc():
    if len(tics)==0:
        return None
    else:
        return time.time()-tics.pop()

def d(tensor=None):
    """
    Returns a device string either for the best available device,
    or for the device corresponding to the argument
    :param tensor:
    :return:
    """
    if tensor is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if type(tensor) == bool:
        return 'cuda'if tensor else 'cpu'
    return 'cuda' if tensor.is_cuda else 'cpu'

def here(subpath=None):
    """
    :return: the path in which the package resides (the directory containing the 'kgbench' dir)
    """
    if subpath is None:
        return os.path.abspath(os.path.dirname(__file__))

    return os.path.abspath(os.path.join(os.path.dirname(__file__), subpath))

def load_rdf(rdf_file, name='', format='nt', store_file='./cache'):
    """
    Load an RDF file into a persistent store, creating the store if necessary.

    If the store exists, return the stored graph.

    :param file:
    :param store_name:
    :return:
    """
    if store_file is None:
        # use in-memory store
        graph = rdf.Graph()
        graph.parse(rdf_file, format=format)
        return graph

    graph = rdf.Graph(store='Sleepycat', identifier=f'kgbench-{name}')
    rt = graph.open(store_file + '-' + name, create=False)

    if rt == rdf.store.NO_STORE:
        print('Persistent store not found. Loading data.')
        rt = graph.open(store_file, create=True)
        graph.parse(rdf_file, format=format)

    else:
        assert rt == rdf.store.VALID_STORE, "The underlying store is corrupt"

        print('Persistent store exists. Loading.')

    return graph




