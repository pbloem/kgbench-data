import kgbench as kg
import numpy as np
import pandas as pd

"""
Generate canonical splits for link prediction.

Generates a single column of integers, one per triple in the data that serves as a mask. The integer values have the 
following meaning:
    0 : training
    1 : validation
    2 : testing
    3 : meta testing
"""

for name in ['amplus', 'dmgfull', 'dmg832k', 'dblp', 'mdgenre']:
    data = kg.load(name)

    nt = data.triples.shape[0]

    # fixed seed for deterministic output
    np.random.seed(0)

    meta_size = 20_000
    test_size = 20_000
    val_size = 20_000
    train_size = nt - test_size - val_size - meta_size

    assert train_size > 0

    print(f'train {train_size}, val {val_size}, test {test_size}, meta {meta_size}')

    bin = np.concatenate([
        np.full((train_size,), 0),
        np.full((val_size,), 1),
        np.full((test_size,), 2),
        np.full((meta_size,), 3)], axis=0)

    np.random.shuffle(bin)  # in place

    bin = pd.DataFrame(data=bin, columns=['in_dataset'])
    bin.to_csv(name + '/' + 'linkprediction-split.csv', index=False, header=None)


