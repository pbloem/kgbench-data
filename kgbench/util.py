import os
import torch

import rdflib as rdf

import time
import PIL
from PIL import ImageOps
import numpy as np

import torchvision as tv

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

def to_tvbatches(images, batch_size=16,  min_size=0, dtype=None, prep=tv.transforms.ToTensor()):
    """
    Returns a generator over torch batches of tensors, using torchvision transforms to translate from
    PIL images to tensors.

    :param images:
    :param torch:
    :return:
    """

    batches = []
    for fr in range(0, len(images), batch_size):
        batch = images[fr:fr+batch_size]

        yield to_tvbatch(batch, min_size=min_size, dtype=dtype, prep=prep)

def to_tensorbatches(images, batch_size=16, use_torch=False, min_size=0, dtype=None):
    """
    Returns a generator over batches of tensors.

    :param images:
    :param torch:
    :return:
    """

    batches = []
    for fr in range(0, len(images), batch_size):
        batch = images[fr:fr+batch_size]

        yield to_tensorbatch(batch, use_torch, min_size, dtype)

def to_tensorbatch(images, use_torch=False, min_size=0, dtype=None):

    maxw = max(max([img.size[0] for img in images]), min_size)
    maxh = max(max([img.size[1] for img in images]), min_size)

    res = []
    for img in images:
        img = pad(img, (maxw, maxh))
        img = np.array(img)[None, :, :, :].astype(np.double)/255.
        res.append( img.transpose((0, 3, 1, 2)) ) # bchw dim ordering

    res = np.concatenate(res, axis=0)
    res = torch.from_numpy(res) if use_torch else res
    if dtype is not None:
        res = res.to(dtype) if use_torch else res.astype(dtype)
    return res

def to_tvbatch(images, min_size=0, dtype=None, prep=tv.transforms.ToTensor()):

    maxw = max(max([img.size[0] for img in images]), min_size)
    maxh = max(max([img.size[1] for img in images]), min_size)

    res = []
    for img in images:
        img = pad(img, (maxw, maxh))
        res.append(prep(img)[None, :, :, :])

    res = torch.cat(res, dim=0)
    if dtype is not None:
        res = res.to(dtype)

    return res

def pad(im, desired_size):

    dw = desired_size[0] - im.size[0]
    dh = desired_size[1] - im.size[1]
    padding = (dw // 2, dh // 2, dw - (dw // 2), dh - (dh // 2))

    return ImageOps.expand(im, padding)



