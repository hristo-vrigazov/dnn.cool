from dnn_cool.task_flow import TaskFlow, NestedResult, BinaryClassificationTask, ClassificationTask, \
    BinaryHardcodedTask, LocalizationTask, NestedClassificationTask


def test_example_flow():

    class CarsBgNetFlow(TaskFlow):

        def __init__(self):
            super().__init__(
                name='carsbg_flow',
                tasks=[
                    BinaryHardcodedTask(name='sliced'),
                    LocalizationTask(name='car_localization'),
                    LocalizationTask(name='license_plate_localization'),
                    BinaryClassificationTask(name='is_car'),
                    BinaryClassificationTask(name='has_lp'),
                    ClassificationTask(name='brand'),
                    ClassificationTask(name='color'),
                    ClassificationTask(name='year'),
                    NestedClassificationTask(name='model', top_k=5)
                ])

        def flow(self, x: NestedResult) -> NestedResult:
            out = NestedResult(self)
            out += self.sliced(x)
            out += self.car_localization(x.car) | (~out.sliced)

            out += self.is_car(x.car, out.sliced)

            out += self.brand(x.common) | out.is_car
            out += self.color(x.common) | out.is_car
            out += self.year(x.common) | out.is_car

            out += self.model(x.common, out.brand) | out.is_car

            out += self.has_lp(x.lp) | out.is_car
            out += self.license_plate_localization(x.lp) | (out.has_lp & out.is_car)

            return out

    carsbg = CarsBgNetFlow()
    task_input = NestedResult(carsbg)
    task_input.res['car'] = 'car'
    task_input.res['common'] = 'common'
    task_input.res['lp'] = 'lp'
    r = carsbg.flow(task_input)

    print('Final res')
    print(r)
