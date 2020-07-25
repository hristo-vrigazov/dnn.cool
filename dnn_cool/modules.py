from typing import Dict

from torch import nn


class SigmoidAndMSELoss(nn.Module):

    def __init__(self, reduction):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(self, output, target):
        activated_output = self.sigmoid(output)
        return self.mse(activated_output, target)


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


# TODO: is there a way to vectorize this in case all counts are different?
class NestedFC(nn.Module):

    def __init__(self, in_features, out_features_nested, bias, top_k):
        super().__init__()
        fcs = []
        for i, out_features in enumerate(out_features_nested):
            fcs.append(nn.Linear(in_features, out_features, bias))
        self.fcs = nn.ModuleList(fcs)
        self.top_k = top_k

    def forward(self, features, parent_flow_dict):
        """
        features and parent_indices must have the same length. Iterates over parent indices, and records predictions
        for every pair (N, P), where N is a batch number and P is a parent index. returns list of lists.
        :param features:
        :param parent_indices: FlowDict which holds the results
        :return: list of lists of lists, where every element is the prediction of the respective module. The first
        len is equal to the batch size, the second len is equal to the top_k and the third len is equal to the respective
        number of classes for the child FC.
        """
        n = len(features)
        parent_indices = parent_flow_dict.decoded[:, :self.top_k]
        res = []
        for i in range(n):
            res_for_parent = []
            for parent_index in parent_indices[i]:
                res_for_parent.append(self.fcs[parent_index](features[i:i+1]))
            res.append(res_for_parent)
        return res


def find_arg_with_gt(args, is_kwargs):
    new_args = {} if is_kwargs else []
    gt = None
    for arg in args:
        if is_kwargs:
            arg = args[arg]
        try:
            gt = arg['gt']
            new_args.append(arg['value'])
        except:
            new_args.append(arg)
    return gt, new_args


def select_gt(a_gt, k_gt):
    if a_gt is None and k_gt is None:
        return {}

    if a_gt is None:
        return k_gt

    return a_gt


def find_gt_and_process_args_when_training(*args, **kwargs):
    a_gt, args = find_arg_with_gt(args, is_kwargs=False)
    k_gt, kwargs = find_arg_with_gt(kwargs, is_kwargs=True)
    return args, kwargs, select_gt(a_gt, k_gt)


class FlowDictDecorator(nn.Module):

    def __init__(self, task):
        super().__init__()
        self.key = task.get_name()
        self.module = task.torch()
        self.activation = task.get_activation()
        self.decoder = task.get_decoder()

    def forward(self, *args, **kwargs):
        if self.training:
            args, kwargs, gt = find_gt_and_process_args_when_training(*args, **kwargs)
            decoded_logits = gt.get(self.key, None)
        else:
            decoded_logits = None

        logits = self.module(*args, **kwargs)
        activated_logits = self.activation(logits) if self.activation is not None else logits

        if decoded_logits is None:
            decoded_logits = self.decoder(activated_logits) if self.decoder is not None else activated_logits

        return FlowDict({
            self.key: FlowDict({
                'logits': logits,
                'activated': activated_logits,
                'decoded': decoded_logits
            })
        })


class TaskFlowModule(nn.Module):

    def __init__(self, task_flow):
        super().__init__()
        self._task_flow = task_flow
        # Save a reference to the flow function of the original class
        # We will then call it by replacing the self, this way effectively running
        # it with this class. And this class stores Pytorch modules as class attributes
        self.flow = task_flow.get_flow_func()

        for key, task in task_flow.tasks.items():
            setattr(self, key, FlowDictDecorator(task))

    def forward(self, x):
        x['training'] = self.training
        flow_dict_res = self.flow(self, FlowDict(x), FlowDict({}))
        return flow_dict_res.flatten()


class FlowDict:

    def __init__(self, res: Dict):
        self.res = res
        self.preconditions = {}

    def __getattr__(self, attr):
        if not ('training' in self.res):
            return self.res[attr]
        if not self.res['training']:
            return self.res[attr]
        return {
            'value': self.res[attr],
            'gt': self.res['gt'],
        }

    def __or__(self, result):
        """
        Note that this has the meaning of preconditioning, not boolean __or__. If you want to use boolean or,
        use the `or_else` method.
        :param result:
        :return:
        """
        for key in self.res:
            current_precondition = self.preconditions.get(key, result.decoded)
            self.preconditions[key] = current_precondition & result.decoded
        return self

    def __invert__(self):
        res = {
            'decoded': ~self.decoded
        }
        return self.__shallow_copy_keys(res)

    def __and__(self, other):
        res = {
            'decoded': self.decoded & other.decoded
        }
        return self.__shallow_copy_keys(res)

    def __getitem__(self, key):
        return self.res[key]

    def __setitem__(self, key, item):
        self.res[key] = item

    def __iter__(self):
        return self.res.__iter__()

    def __contains__(self, item):
        return self.res.__contains__(item)

    def __shallow_copy_keys(self, res):
        for key, value in self.res.items():
            if key not in res:
                res[key] = value
        return FlowDict(res)

    def or_else(self, other):
        res = {
            'decoded': self.decoded | other.decoded
        }
        return self.__shallow_copy_keys(res)

    def __xor__(self, other):
        res = {
            'decoded': self.decoded ^ other.decoded
        }
        return self.__shallow_copy_keys(res)

    def __sub__(self, other):
        res = FlowDict({})
        for key, value in self.res.items():
            if key not in other.res:
                res.res[key] = value
        for key, value in self.preconditions.items():
            if key not in other.preconditions:
                res.preconditions[key] = value
        return res

    def __add__(self, other):
        res = FlowDict({})
        res.res.update(self.res)
        res.res.update(other.res)
        res.preconditions.update(self.preconditions)
        res.preconditions.update(other.preconditions)
        return res

    def __iadd__(self, other):
        self.res.update(other.res)
        self.preconditions.update(other.preconditions)
        return self

    def flatten(self):
        res = {}
        for key, value in self.res.items():
            keys, values = self._traverse(key, value, self.preconditions.get(key, None))
            for i in range(len(keys)):
                res[keys[i]] = values[i]

        return res

    def _traverse(self, parent_key, parent_value, parent_precondition):
        is_leaf = not isinstance(parent_value.logits, dict)
        if is_leaf:
            res_keys = [parent_key]
            res_values = [parent_value.logits]
            if parent_precondition is not None:
                res_keys.append(f'precondition|{parent_key}')
                res_values.append(parent_precondition)
            return res_keys, res_values

        all_keys, all_values = [], []
        for key, value in parent_value.logits.items():
            if key.startswith('precondition|'):
                continue
            full_path_key = f'{parent_key}.{key}'
            all_keys.append(full_path_key)
            all_values.append(value)
            # if there is a parent precondition, then the child has to satisfy both his own and his parents'
            # preconditions.
            current_precondition = parent_value.logits.get(f'precondition|{key}', parent_precondition)
            if current_precondition is None:
                current_precondition = parent_precondition
            elif parent_precondition is not None:
                current_precondition &= parent_precondition
            all_keys.append(f'precondition|{full_path_key}')
            all_values.append(current_precondition)
        return all_keys, all_values
