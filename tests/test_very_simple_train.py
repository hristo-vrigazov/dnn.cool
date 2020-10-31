from collections import OrderedDict

import numpy as np
import pandas as pd
import torch

from dnn_cool.catalyst_utils import InterpretationCallback, TensorboardConverters, TensorboardConverter, \
    ReplaceGatherCallback, img_publisher, text_publisher
from dnn_cool.converters import Converters
from dnn_cool.project import Project
from dnn_cool.runner import InferDictCallback
from dnn_cool.synthetic_dataset import synthetic_dataset_preparation
from dnn_cool.task_flow import TaskFlow, BinaryClassificationTask, ClassificationTask


def test_project_example():
    df_data = [
        {'camera_blocked': True, 'door_open': True, 'uniform_type': 0, 'input': 0},
        {'camera_blocked': False, 'door_open': True, 'uniform_type': 1, 'input': 1},
        {'camera_blocked': False, 'door_open': True, 'uniform_type': 0, 'input': 2},
        {'camera_blocked': False, 'door_open': True, 'uniform_type': 2, 'input': 3},
        {'camera_blocked': True, 'door_open': True, 'uniform_type': 1, 'input': 4},
        {'camera_blocked': True, 'door_open': True, 'uniform_type': 0, 'input': 0},
        {'camera_blocked': False, 'door_open': True, 'uniform_type': 1, 'input': 1},
        {'camera_blocked': False, 'door_open': True, 'uniform_type': 0, 'input': 2},
        {'camera_blocked': False, 'door_open': True, 'uniform_type': 2, 'input': 3},
        {'camera_blocked': True, 'door_open': True, 'uniform_type': 1, 'input': 4},
    ]

    df = pd.DataFrame(df_data)

    converters = Converters()
    converters.values.type_mapping['category'] = torch.LongTensor
    converters.values.type_mapping['binary'] = torch.BoolTensor

    converters.task.type_mapping['binary'] = BinaryClassificationTask
    converters.task.type_mapping['category'] = ClassificationTask

    project = Project(df, input_col='input',
                      output_col=['camera_blocked', 'door_open', 'uniform_type'],
                      project_dir='./example_project',
                      converters=converters)

    @project.add_flow
    def camera_not_blocked_flow(flow, x, out):
        out += flow.door_open(x.features)
        out += flow.uniform_type(x.features) | out.door_open
        return out

    @project.add_flow
    def all_pipeline(flow, x, out):
        out += flow.camera_blocked(x.features)
        out += flow.camera_not_blocked_flow(x.features) | out.camera_blocked
        return out

    flow: TaskFlow = project.get_full_flow()
    print(flow)

    dataset = flow.get_dataset()
    print(dataset[0])


def test_inference_synthetic_treelib(treelib_explanation_on_first_batch):
    treelib_explanation_on_first_batch.show()


def test_interpretation_synthetic():
    model, nested_loaders, datasets, project = synthetic_dataset_preparation()
    runner = project.runner(model=model, runner_name='default_experiment')
    flow = project.get_full_flow()

    loaders = OrderedDict({'infer': nested_loaders['valid']})

    model = runner.best()

    tensorboard_converters = TensorboardConverters(
        logdir=runner.project_dir / runner.default_logdir,
        tensorboard_loggers=TensorboardConverter(),
        datasets=datasets
    )

    infer_dict_callback = InferDictCallback()
    callbacks = OrderedDict([
        ("interpretation", InterpretationCallback(flow, tensorboard_converters)),
        ("inference", infer_dict_callback),
        ("reducer", ReplaceGatherCallback(flow, infer_dict_callback))
    ])
    r = runner.infer(loaders=loaders, callbacks=callbacks)

    interpretations = callbacks["interpretation"].interpretations
    print(interpretations)


def test_interpretation_default_runner():
    model, nested_loaders, datasets, project = synthetic_dataset_preparation()
    runner = project.runner(model=model, runner_name='default_experiment')
    model = runner.best()
    project.converters.tensorboard_converters.type_mapping['img'] = [img_publisher]
    project.converters.tensorboard_converters.type_mapping['text'] = [text_publisher]
    r = runner.infer(model=model)
    print(r)


def test_tune_pipeline():
    model, nested_loaders, datasets, project = synthetic_dataset_preparation()
    runner = project.runner(model=model, runner_name='default_experiment')
    tuned_params = runner.tune()
    print(tuned_params)


def test_load_tuned_pipeline():
    model, nested_loaders, datasets, project = synthetic_dataset_preparation()
    runner = project.runner(model=model, runner_name='default_experiment')
    tuned_params = runner.load_tuned()
    print(tuned_params)


def test_load_tuned_pipeline_from_decoder():
    model, nested_loaders, datasets, project = synthetic_dataset_preparation()
    runner = project.runner(model=model, runner_name='default_experiment')
    tuned_params = torch.load(runner.project_dir / runner.default_logdir / 'tuned_params.pkl')
    flow = project.get_full_flow()
    flow.get_decoder().load_tuned(tuned_params)


def test_evaluation_is_shown():
    model, nested_loaders, datasets, project = synthetic_dataset_preparation()
    runner = project.runner(model=model, runner_name='default_experiment')
    evaluation = runner.evaluate()
    accuracy_df = evaluation[evaluation['metric_name'] == 'accuracy']
    assert np.alltrue(accuracy_df['metric_res'] > 0.98)
    mae_df = evaluation[evaluation['metric_name'] == 'mean_absolute_error']
    assert np.alltrue(mae_df['metric_res'] < 5e-2)
    pd.set_option('display.max_columns', None)


def test_composite_activation():
    model, nested_loaders, datasets, project = synthetic_dataset_preparation()
    runner = project.runner(model=model, runner_name='default_experiment')
    flow = project.get_full_flow()
    activation = flow.get_activation()
    res = runner.load_inference_results()
    activated_predictions = activation(res['logits']['test'])
    print(activated_predictions)


def test_composite_decoding():
    model, nested_loaders, datasets, project = synthetic_dataset_preparation()
    runner = project.runner(model=model, runner_name='default_experiment')
    flow = project.get_full_flow()
    decoder = flow.get_decoder()
    res = runner.load_inference_results()
    activated_predictions = decoder(res['logits']['test'])
    print(activated_predictions)


def test_composite_filtering():
    model, nested_loaders, datasets, project = synthetic_dataset_preparation()
    runner = project.runner(model=model, runner_name='default_experiment')
    flow = project.get_full_flow()
    filter_func = flow.get_filter()
    res = runner.load_inference_results()
    filtered_results = filter_func(res['logits']['test'], res['targets']['test'])
    print(filtered_results)

