from collections import OrderedDict

import numpy as np
import pandas as pd
import torch

from dnn_cool.catalyst_utils import InterpretationCallback, TensorboardConverters, TensorboardConverter, \
    ReplaceGatherCallback, img_publisher, text_publisher
from dnn_cool.runner import InferDictCallback, DnnCoolSupervisedRunner, DnnCoolRunnerView
from dnn_cool.synthetic_dataset import synthetic_dataset_preparation, get_synthetic_full_flow, SecurityModule


def test_inference_synthetic_treelib(treelib_explanation_on_first_batch):
    treelib_explanation_on_first_batch.show()


def test_interpretation_synthetic():
    model, nested_loaders, datasets, full_flow_for_development, tensorboard_converters = synthetic_dataset_preparation()
    runner = DnnCoolSupervisedRunner(model=model,
                                     full_flow=full_flow_for_development,
                                     project_dir='./security_project',
                                     runner_name='default_experiment',
                                     tensoboard_converters=tensorboard_converters,
                                     balance_dataparallel_memory=True)
    loaders = OrderedDict({'infer': nested_loaders['valid']})
    model = runner.best()

    tensorboard_converters = TensorboardConverters(
        logdir=runner.project_dir / runner.default_logdir,
        tensorboard_loggers=TensorboardConverter(),
        datasets=datasets
    )

    infer_dict_callback = InferDictCallback()
    callbacks = OrderedDict([
        ("interpretation", InterpretationCallback(full_flow_for_development.get_per_sample_criterion(),
                                                  tensorboard_converters=tensorboard_converters)),
        ("inference", infer_dict_callback),
        ("reducer", ReplaceGatherCallback(full_flow_for_development, infer_dict_callback))
    ])
    r = runner.infer(loaders=loaders, callbacks=callbacks)

    interpretations = callbacks["interpretation"].interpretations
    print(interpretations)


def test_interpretation_default_runner():
    model, nested_loaders, datasets, full_flow_for_development, tensorboard_converters = synthetic_dataset_preparation()
    runner = DnnCoolSupervisedRunner(model=model,
                                     full_flow=full_flow_for_development,
                                     project_dir='./security_project',
                                     runner_name='default_experiment',
                                     tensoboard_converters=tensorboard_converters,
                                     balance_dataparallel_memory=True)
    model = runner.best()
    tensorboard_converters.type_mapping['img'] = [img_publisher]
    tensorboard_converters.type_mapping['text'] = [text_publisher]
    r = runner.infer(model=model)
    print(r)


def test_tune_pipeline():
    model, nested_loaders, datasets, full_flow_for_development, tensorboard_converters = synthetic_dataset_preparation()
    runner = DnnCoolSupervisedRunner(model=model,
                                     full_flow=full_flow_for_development,
                                     project_dir='./security_project',
                                     runner_name='default_experiment',
                                     tensoboard_converters=tensorboard_converters,
                                     balance_dataparallel_memory=True)
    tuned_params = runner.tune()
    print(tuned_params)


def test_load_tuned_pipeline():
    model, nested_loaders, datasets, full_flow_for_development, tensorboard_converters = synthetic_dataset_preparation()
    runner = DnnCoolSupervisedRunner(model=model,
                                     full_flow=full_flow_for_development,
                                     project_dir='./security_project',
                                     runner_name='default_experiment',
                                     tensoboard_converters=tensorboard_converters,
                                     balance_dataparallel_memory=True)
    tuned_params = runner.load_tuned()
    print(tuned_params)


def test_load_tuned_pipeline_from_decoder():
    model, nested_loaders, datasets, full_flow_for_development, tensorboard_converters = synthetic_dataset_preparation()
    runner = DnnCoolSupervisedRunner(model=model,
                                     full_flow=full_flow_for_development,
                                     project_dir='./security_project',
                                     runner_name='default_experiment',
                                     tensoboard_converters=tensorboard_converters,
                                     balance_dataparallel_memory=True)
    tuned_params = torch.load(runner.project_dir / runner.default_logdir / 'tuned_params.pkl')
    full_flow_for_development.task.get_decoder().load_tuned(tuned_params)


def test_evaluation_is_shown():
    model, nested_loaders, datasets, full_flow_for_development, tensorboard_converters = synthetic_dataset_preparation()
    runner = DnnCoolSupervisedRunner(model=model,
                                     full_flow=full_flow_for_development,
                                     project_dir='./security_project',
                                     runner_name='default_experiment',
                                     tensoboard_converters=tensorboard_converters,
                                     balance_dataparallel_memory=True)
    evaluation = runner.evaluate()
    accuracy_df = evaluation[evaluation['metric_name'] == 'accuracy']
    assert np.alltrue(accuracy_df['metric_res'] > 0.98)
    mae_df = evaluation[evaluation['metric_name'] == 'mean_absolute_error']
    assert np.alltrue(mae_df['metric_res'] < 5e-2)
    pd.set_option('display.max_columns', None)


def test_composite_activation():
    model, nested_loaders, datasets, full_flow_for_development, tensorboard_converters = synthetic_dataset_preparation()
    runner = DnnCoolSupervisedRunner(model=model,
                                     full_flow=full_flow_for_development,
                                     project_dir='./security_project',
                                     runner_name='default_experiment',
                                     tensoboard_converters=tensorboard_converters,
                                     balance_dataparallel_memory=True)
    activation = full_flow_for_development.task.get_activation()
    res = runner.load_inference_results()
    activated_predictions = activation(res['logits']['test'])
    print(activated_predictions)


def test_composite_decoding():
    model, nested_loaders, datasets, full_flow_for_development, tensorboard_converters = synthetic_dataset_preparation()
    runner = DnnCoolSupervisedRunner(model=model,
                                     full_flow=full_flow_for_development,
                                     project_dir='./security_project',
                                     runner_name='default_experiment',
                                     tensoboard_converters=tensorboard_converters,
                                     balance_dataparallel_memory=True)
    decoder = full_flow_for_development.task.get_decoder()
    res = runner.load_inference_results()
    activated_predictions = decoder(res['logits']['test'])
    print(activated_predictions)


def test_composite_filtering():
    model, nested_loaders, datasets, full_flow_for_development, tensorboard_converters = synthetic_dataset_preparation()
    runner = DnnCoolSupervisedRunner(model=model,
                                     full_flow=full_flow_for_development,
                                     project_dir='./security_project',
                                     runner_name='default_experiment',
                                     tensoboard_converters=tensorboard_converters,
                                     balance_dataparallel_memory=True)
    filter_func = full_flow_for_development.get_filter()
    res = runner.load_inference_results()
    filtered_results = filter_func(res['logits']['test'], res['targets']['test'])
    print(filtered_results)


def test_global_indices_conversion():
    full_flow = get_synthetic_full_flow(n_shirt_types=7, n_facial_characteristics=3)
    model = SecurityModule(full_flow)
    runner = DnnCoolRunnerView(full_flow=full_flow, model=model,
                               project_dir='./security_project', runner_name='default_experiment')
    global_idx, r = (runner.worst_examples('test', 'person_regression.face_regression.face_y1', 10))

    mae_from_interpretations = (r['activated'] - r['targets']).abs().mean().item()
    task_mask = runner.evaluation_df['task_path'] == 'person_regression.face_regression.face_y1'
    metric_mask = runner.evaluation_df['metric_name'] == 'mean_absolute_error'
    sub_df = runner.evaluation_df[task_mask & metric_mask]

    assert abs(mae_from_interpretations - sub_df.iloc[0]['metric_res']) < 1e6