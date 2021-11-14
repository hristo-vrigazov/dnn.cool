
def initialize_dict_structure(dct):
    res = {}
    for key, value in dct.items():
        if isinstance(value, dict):
            res[key] = initialize_dict_structure(value)
            continue
        res[key] = []
    return res


def find_max_len_of_list_of_lists(ll):
    r = [max(len(l) for l in ll)]
    tmp = ll
    child = tmp[0]
    while isinstance(child, list):
        r.append(max(len(l) for l in child))
        tmp = child
        child = tmp[0]
    return r


def find_padding_shape_of_nested_list(ll):
    return [len(ll)] + find_max_len_of_list_of_lists(ll)


def append_example_to_dict(dct, shapes, ex):
    for key, value in ex.items():
        if isinstance(value, dict):
            dct[key] = append_example_to_dict(dct[key], shapes[key], value)
            continue
        dct[key].append(value)
    return dct


def examples_to_nested_list(examples):
    X_ex, y_ex = examples[0]
    X_batch = initialize_dict_structure(X_ex)
    y_batch = initialize_dict_structure(y_ex)

    X_shapes = initialize_dict_structure(X_ex)
    y_shapes = initialize_dict_structure(y_ex)

    for example in examples:
        X_ex, y_ex = example
