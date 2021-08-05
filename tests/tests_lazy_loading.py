from time import time

from dnn_cool.synthetic_dataset import synthetic_dataset_preparation


def test_lazy_loading_project():
    start = time()
    model, nested_loaders, datasets, project = synthetic_dataset_preparation(perform_conversion=False)
    runner = project.runner(model=model, runner_name='security_logs')
    model = runner.best()
    print(model)
    print(time() - start)
