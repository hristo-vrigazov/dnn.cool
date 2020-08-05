

def positive_values(tensor):
    return tensor >= 0.


def positive_values_unsqueezed(tensor):
    return (tensor >= 0.).unsqueeze(dim=-1)
