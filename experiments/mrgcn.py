"""
Run the MRGCN baseline of the given dataset.

This is a minimal version of the MRGCN model (). It passes the representation of each literal datatype through a fixed
pre-trained encoder (BERT for strings, ResNet-18 for images). These encodings are fixed and not trained. We then
transform each instance to a fixed embedding space by a single linear transformation for each datatype. The uris and
blank nodes are given fixed embeddings.

"""

import fire, sys, tqdm

import torch
from torch import nn
import torch.nn.functional as F
from kgbench import load, tic, toc, d
import kgbench as kg

from collections import Counter

import transformers as tf
from torchvision import transforms

from sklearn.decomposition import PCA

def enrich(triples : torch.Tensor, n : int, r: int):

    cuda = triples.is_cuda

    inverses = torch.cat([
        triples[:, 2:],
        triples[:, 1:2] + r,
        triples[:, :1]
    ], dim=1)

    selfloops = torch.cat([
        torch.arange(n, dtype=torch.long,  device=d(cuda))[:, None],
        torch.full((n, 1), fill_value=2*r),
        torch.arange(n, dtype=torch.long, device=d(cuda))[:, None],
    ], dim=1)

    return torch.cat([triples, inverses, selfloops], dim=0)

def sum_sparse(indices, values, size, row=True):
    """
    Sum the rows or columns of a sparse matrix, and redistribute the
    results back to the non-sparse row/column entries

    :return:
    """

    ST = torch.cuda.sparse.FloatTensor if indices.is_cuda else torch.sparse.FloatTensor

    assert len(indices.size()) == 2

    k, r = indices.size()

    if not row:
        # transpose the matrix
        indices = torch.cat([indices[:, 1:2], indices[:, 0:1]], dim=1)
        size = size[1], size[0]

    ones = torch.ones((size[1], 1), device=d(indices))

    smatrix = ST(indices.t(), values, size=size)
    sums = torch.mm(smatrix, ones) # row/column sums

    sums = sums[indices[:, 0]]

    assert sums.size() == (k, 1)

    return sums.view(k)

def adj(triples, num_nodes, num_rels, cuda=False, vertical=True):
    """
     Computes a sparse adjacency matrix for the given graph (the adjacency matrices of all
     relations are stacked vertically).

     :param edges: List representing the triples
     :param i2r: list of relations
     :param i2n: list of nodes
     :return: sparse tensor
    """
    r, n = num_rels, num_nodes
    size = (r * n, n) if vertical else (n, r * n)

    from_indices = []
    upto_indices = []

    for fr, rel, to in triples:

        offset = rel.item() * n

        if vertical:
            fr = offset + fr.item()
        else:
            to = offset + to.item()

        from_indices.append(fr)
        upto_indices.append(to)

    indices = torch.tensor([from_indices, upto_indices], dtype=torch.long, device=d(cuda))

    assert indices.size(1) == len(triples)
    assert indices[0, :].max() < size[0], f'{indices[0, :].max()}, {size}, {r}'
    assert indices[1, :].max() < size[1], f'{indices[1, :].max()}, {size}, {r}'

    return indices.t(), size

class RGCN(nn.Module):
    """
    We use a classic RGCN, with embeddings as inputs (instead of the one-hot inputs of rgcn.py)

    """

    def __init__(self, triples, n, r, insize, hidden, numcls, bases=None):

        super().__init__()

        self.insize = insize
        self.hidden = hidden
        self.bases = bases
        self.numcls = numcls

        self.triples = enrich(triples, n, r)

        # horizontally and vertically stacked versions of the adjacency graph
        hor_ind, hor_size = adj(self.triples, n, 2*r+1, vertical=False)
        ver_ind, ver_size = adj(self.triples, n, 2*r+1, vertical=True)

        _, rn = hor_size
        r = rn // n

        vals = torch.ones(ver_ind.size(0), dtype=torch.float)
        vals = vals / sum_sparse(ver_ind, vals, ver_size)

        hor_graph = torch.sparse.FloatTensor(indices=hor_ind.t(), values=vals, size=hor_size)
        self.register_buffer('hor_graph', hor_graph)

        ver_graph = torch.sparse.FloatTensor(indices=ver_ind.t(), values=vals, size=ver_size)
        self.register_buffer('ver_graph', ver_graph)

        # layer 1 weights
        if bases is None:
            self.weights1 = nn.Parameter(torch.FloatTensor(r, insize, hidden))
            nn.init.xavier_uniform_(self.weights1, gain=nn.init.calculate_gain('relu'))

            self.bases1 = None
        else:
            self.comps1 = nn.Parameter(torch.FloatTensor(r, bases))
            nn.init.xavier_uniform_(self.comps1, gain=nn.init.calculate_gain('relu'))

            self.bases1 = nn.Parameter(torch.FloatTensor(bases, insize, hidden))
            nn.init.xavier_uniform_(self.bases1, gain=nn.init.calculate_gain('relu'))

        # layer 2 weights
        if bases is None:

            self.weights2 = nn.Parameter(torch.FloatTensor(r, hidden, numcls) )
            nn.init.xavier_uniform_(self.weights2, gain=nn.init.calculate_gain('relu'))

            self.bases2 = None
        else:
            self.comps2 = nn.Parameter(torch.FloatTensor(r, bases))
            nn.init.xavier_uniform_(self.comps2, gain=nn.init.calculate_gain('relu'))

            self.bases2 = nn.Parameter(torch.FloatTensor(bases, hidden, numcls))
            nn.init.xavier_uniform_(self.bases2, gain=nn.init.calculate_gain('relu'))

        self.bias1 = nn.Parameter(torch.FloatTensor(hidden).zero_())
        self.bias2 = nn.Parameter(torch.FloatTensor(numcls).zero_())

    def forward(self, features):

        # size of node representation per layer: f -> e -> c
        n, rn = self.hor_graph.size()
        r = rn // n
        e = self.hidden
        b, c = self.bases, self.numcls

        n, f = features.size()

        ## Layer 1
        h = torch.mm(self.ver_graph, features) # sparse mm
        h = h.view(r, n, f) # new dim for the relations

        if self.bases1 is not None:
            weights = torch.einsum('rb, bij -> rij', self.comps1, self.bases1)
            # weights = torch.mm(self.comps1, self.bases1.view(b, n*e)).view(r, n, e)
        else:
            weights = self.weights1

        assert weights.size() == (r, f, e)

        # Apply weights and sum over relations
        h = torch.bmm(h, weights).sum(dim=0)

        assert h.size() == (n, e)

        h = F.relu(h + self.bias1)

        ## Layer 2

        # Multiply adjacencies by hidden
        h = torch.mm(self.ver_graph, h) # sparse mm
        h = h.view(r, n, e) # new dim for the relations

        if self.bases2 is not None:
            weights = torch.einsum('rb, bij -> rij', self.comps2, self.bases2)
            # weights = torch.mm(self.comps2, self.bases2.view(b, e * c)).view(r, e, c)
        else:
            weights = self.weights2

        # Apply weights, sum over relations
        # h = torch.einsum('rhc, rnh -> nc', weights, h)
        h = torch.bmm(h, weights).sum(dim=0)

        assert h.size() == (n, c)

        return h + self.bias2 # -- softmax is applied in the loss

    def penalty(self, p=2):

        assert p==2

        if self.bases is None:
            return self.weights1.pow(2).sum()

        return self.comps1.pow(p).sum() + self.bases1.pow(p).sum()

def mobilenet_emb(pilimages, bs=512):

    # Create embeddings for image
    prep = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    #-- Standard mobilenet preprocessing.

    image_embeddings = []

    model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)

    if torch.cuda.is_available():
        model.cuda()

    nimages = len(pilimages)
    imagegen = kg.to_tvbatches(pilimages, batch_size=bs, prep=prep, min_size=224, dtype=torch.float32)

    for batch in tqdm.tqdm(imagegen, total=nimages // bs):
        bn, c, h, w = batch.size()
        if torch.cuda.is_available():
            batch = batch.cuda()

        out = model.features(batch)
        image_embeddings.append(out.view(bn, -1).to('cpu'))
        # print(image_embeddings[-1].size())

    return torch.cat(image_embeddings, dim=0)

def bert_emb(strings, bs_chars, mname='distilbert-base-cased'):
    # Sort by length and reverse the sort after computing embeddings
    # (this will speed up computation of the embeddings, by reducing the amount of padding required)

    indexed = list(enumerate(strings))
    indexed.sort(key=lambda p:len(p[1]))

    embeddings = bert_emb_([s for _, s in indexed], bs_chars)
    indices = torch.tensor([i for i, _ in indexed])
    _, iindices = indices.sort()

    return embeddings[iindices]

MNAME='distilbert-base-cased'

bmodel = tf.DistilBertModel.from_pretrained(MNAME)
btok = tf.DistilBertTokenizerFast.from_pretrained(MNAME)

def bert_emb_(strings, bs_chars, ):

    pbar = tqdm.tqdm(total=len(strings))

    outs = []
    fr = 0
    while fr < len(strings):

        to = fr
        bs = 0
        while bs < bs_chars and to < len(strings):
            bs += len(strings[to])
            to += 1
            # -- add strings to the batch until it puts us over bs_chars

        # print('batch', fr, to, len(strings))
        strbatch = strings[fr:to]

        try:
            batch = btok(strbatch, padding=True, truncation=True, return_tensors="pt")
        except:
            print(strbatch)
            sys.exit()
        #-- tokenizer automatically prepends the CLS token
        inputs, mask = batch['input_ids'], batch['attention_mask']
        if torch.cuda.is_available():
            inputs, mask = inputs.cuda(), mask.cuda()

        out = bmodel(inputs, mask)

        outs.append(out[0][:, 0, :].to('cpu')) # use only the CLS token

        pbar.update(len(strbatch))
        fr = to

    return torch.cat(outs, dim=0)

def pca(tensor, target_dim):
    """
    Applies PCA to a torch matrix to reduce it to the target dimension
    """

    n, f = tensor.size()
    if n < 25: # no point in PCA, just clip
       res = tensor[:, :target_dim]
    else:
        if tensor.is_cuda:
            tensor = tensor.to('cpu')
        model = PCA(n_components=target_dim, whiten=True)

        res = model.fit_transform(tensor)
        res =  torch.from_numpy(res)

    if torch.cuda.is_available():
        res = res.cuda()

    return res

def go(name='amplus', lr=0.01, wd=0.0, l2=5e-4, epochs=50, prune=True, optimizer='adam', final=False, emb=16, bases=40, printnorms=None, imagebatch=256, stringbatch=50_000):

    # bert_emb(['.....', '.', '..', '...', '....'], bs_chars = 50_000)

    data = load(name, torch=True, prune_dist=2 if prune else None, final=final)
    data = kg.group(data)

    print(f'{data.triples.size(0)} triples')
    print(f'{data.num_entities} entities')
    print(f'{data.num_relations} relations')

    if torch.cuda.is_available():
        bmodel.cuda()

    tic()
    with torch.no_grad():

        embeddings = []
        for datatype in data.datatypes():
            if datatype in ['iri', 'blank_node']:
                print(f'Initializing embedding for datatype {datatype}.')
                # create random embeddings
                # -- we will parametrize this part of the input later
                n = len(data.get_strings(dtype=datatype))
                nodes = torch.randn(n, emb)
                if torch.cuda.is_available():
                    nodes = nodes.cuda()

                embeddings.append(nodes)

            elif datatype == 'http://kgbench.info/dt#base64Image':
                print(f'Computing embeddings for images.')
                image_embeddings = mobilenet_emb(data.get_images(), bs=imagebatch)
                image_embeddings = pca(image_embeddings, target_dim=emb)
                embeddings.append(image_embeddings)

            else:
                # embed literal strings with DistilBERT
                print(f'Computing embeddings for datatype {datatype}.')
                string_embeddings = bert_emb(data.get_strings(dtype=datatype), bs_chars=stringbatch)
                string_embeddings = pca(string_embeddings, target_dim=emb)
                embeddings.append(string_embeddings)

        embeddings = torch.cat(embeddings, dim=0).to(torch.float)
        # -- note that we use the fact here that the data loader clusters the nodes by data type, in the
        #    order given by data._datasets
    print(f'embeddings created in {toc()} seconds.')

    # Split embeddings into trainable and non-trainable
    num_uri, num_bnode = len(data.datatype_l2g('uri')), len(data.datatype_l2g('blank_node'))
    numparms = num_uri + num_bnode
    trainable = embeddings[:numparms, :]
    constant  = embeddings[numparms:, :]

    trainable = nn.Parameter(trainable)

    tic()
    rgcn = RGCN(data.triples, n=data.num_entities, r=data.num_relations, insize=emb, hidden=emb, numcls=data.num_classes, bases=bases)

    if torch.cuda.is_available():
        print('Using cuda.')
        rgcn.cuda()

        trainable = trainable.cuda()
        constant = constant.cuda()

        data.training = data.training.cuda()
        data.withheld = data.withheld.cuda()

    print(f'construct: {toc():.5}s')

    if optimizer == 'adam':
        opt = torch.optim.Adam(lr=lr, weight_decay=wd, params=rgcn.parameters())
    elif optimizer == 'adamw':
        opt = torch.optim.AdamW(lr=lr, weight_decay=wd, params=rgcn.parameters())
    else:
        raise Exception(f'Optimizer {optimizer} not known')

    for e in range(epochs):
        tic()
        opt.zero_grad()

        features = torch.cat([trainable, constant], dim=0)
        out = rgcn(features)

        idxt, clst = data.training[:, 0], data.training[:, 1]
        idxw, clsw = data.withheld[:, 0], data.withheld[:, 1]

        out_train = out[idxt, :]
        loss = F.cross_entropy(out_train, clst, reduction='mean')
        if l2 != 0.0:
            loss = loss + l2 * rgcn.penalty()

        # compute performance metrics
        with torch.no_grad():
            training_acc = (out[idxt, :].argmax(dim=1) == clst).sum().item() / idxt.size(0)
            withheld_acc = (out[idxw, :].argmax(dim=1) == clsw).sum().item() / idxw.size(0)

        loss.backward()
        opt.step()

        if printnorms is not None:
            # Print relation norms layer 1
            nr = data.num_relations
            weights = rgcn.weights1 if bases is None else rgcn.comps1

            ctr = Counter()

            for r in range(nr):

                ctr[data.i2r[r]] = weights[r].norm()
                ctr['inv_'+ data.i2r[r]] = weights[r+nr].norm()

            print('relations with largest weight norms in layer 1.')
            for rel, w in ctr.most_common(printnorms):
                print(f'     norm {w:.4} for {rel} ')

            weights = rgcn.weights2 if bases is None else rgcn.comps2

            ctr = Counter()
            for r in range(nr):

                ctr[data.i2r[r]] = weights[r].norm()
                ctr['inv_'+ data.i2r[r]] = weights[r+nr].norm()

            print('relations with largest weight norms in layer 2.')
            for rel, w in ctr.most_common(printnorms):
                print(f'     norm {w:.4} for {rel} ')


        print(f'epoch {e:02}: loss {loss:.2}, train acc {training_acc:.2}, \t withheld acc {withheld_acc:.2} \t ({toc():.5}s)')

if __name__ == '__main__':

    print('arguments ', ' '.join(sys.argv))
    fire.Fire(go)