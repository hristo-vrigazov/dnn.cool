from torch import nn

from dnn_cool.utilities import FlowDict


class SigmoidAndMSELoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.mse = nn.MSELoss()

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
        self.activation = task.activation()
        self.decoder = task.decoder()

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
        self.flow = task_flow.__class__.flow

        for key, task in task_flow.tasks.items():
            setattr(self, key, FlowDictDecorator(task))

    def forward(self, x):
        x['training'] = self.training
        return self.flow(self, FlowDict(x), FlowDict({}))
