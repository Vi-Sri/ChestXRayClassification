import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

def split_sampler(n_samples, split):
    if split == 0.0:
        return None, None

    idx_full = np.arange(n_samples)

    np.random.seed(0)
    np.random.shuffle(idx_full)

    if isinstance(split, int):
        assert split > 0
        assert split < n_samples, "validation set size is configured to be larger than entire dataset."
        len_valid = split
    else:
        len_valid = int(n_samples * split)

    valid_idx = idx_full[0:len_valid]
    train_idx = np.delete(idx_full, np.arange(0, len_valid))

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    n_samples = len(train_idx)

    return train_sampler, valid_sampler

t, v = split_sampler(5232, 0.1)
print(len(t.indices))
print(len(v.indices))