import os
import torch

import rdflib as rdf

import time
import PIL
from PIL import ImageOps
import numpy as np


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

def to_tensorbatches(images, batch_size=16, use_torch=False):
    """
    Transforms a list of PIL images to a list of batch tensors. Images are padded to maintain the same size per batch.

    :param images:
    :param torch:
    :return:
    """

    batches = []
    for fr in range(0, len(images), batch_size):
        batch = images[fr:fr+batch_size]

        batches.append(to_tensorbatch(batch, use_torch))

    return batches

def to_tensorbatch(images, use_torch=False, min_size=224):

    b = len(images)
    maxw = max(max([img.size[0] for img in images]), min_size)
    maxh = max(max([img.size[1] for img in images]), min_size)

    res = []
    for img in images:
        img = pad(img, (maxw, maxh))
        img = np.array(img)[None, :, :, :].astype(np.double)/255.
        res.append( img.transpose((0, 3, 1, 2)) ) # bchw dim ordering

    res = np.concatenate(res, axis=0)
    return torch.from_numpy(res).double() if use_torch else res

def pad(im, desired_size):

    dw = desired_size[0] - im.size[0]
    dh = desired_size[1] - im.size[1]
    padding = (dw // 2, dh // 2, dw - (dw // 2), dh - (dh // 2))

    return ImageOps.expand(im, padding)



