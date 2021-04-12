import pandas as pd
import numpy as np
def RandomGroupKFold_split(groups, n, seed=None):  # noqa: N802
    """
    Random analogous of sklearn.model_selection.GroupKFold.split.

    :return: list of (train, test) indices
    """
    groups = pd.Series(groups)
    ix = np.arange(len(groups))
    unique = np.unique(groups)
    np.random.RandomState(seed).shuffle(unique)
    result = []
    for split in np.array_split(unique, n):
        print(split)
        mask = groups.isin(split)
        train, test = ix[~mask], ix[mask]
        result.append((train, test))

    return result