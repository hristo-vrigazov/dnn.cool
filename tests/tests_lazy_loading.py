from time import time

from dnn_cool.runner import DnnCoolRunnerView
from dnn_cool.synthetic_dataset import get_synthetic_full_flow, SecurityModule


def test_lazy_loading_project():
    start = time()
    full_flow = get_synthetic_full_flow(n_shirt_types=7, n_facial_characteristics=3)
    model = SecurityModule(full_flow)
    runner = DnnCoolRunnerView(full_flow=full_flow, model=model,
                               project_dir='./security_project', runner_name='security_logs')
    model = runner.best()
    print(model)
    print(time() - start)
