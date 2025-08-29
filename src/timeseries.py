from itertools import groupby


def sequence_lengths(data):
    """
    identify lengths of inlet closures
    """
    lengths = []
    for key, group in groupby(data):
        group_list = list(group)
        if key == 1:
            lengths.extend([len(group_list)] * len(group_list))
        else:
            lengths.extend([0] * len(group_list))
    return lengths