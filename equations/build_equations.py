
from .arx_equation import ARXEquation


def build_equations(dataset, categorical_idxs, continuous_idxs):
    equations = []
    if dataset == "swat":
        equations.append(ARXEquation(
            [0],
            {0: [111.4, -168.6, 40.37, 26.34, 1.871]},
            [1],
            {1: [1, -1.471, 0.3398, 0.1683, 0.01268, -0.04941]},
            {0: 2.5423,
             1: 538.0311},
            2,
            2,
            categorical_idxs,
            continuous_idxs
        ))
        equations.append(ARXEquation(
            [8, 17],
            {8: [99.47, 44.46, -28.9, -23.7],
             17: [9.959, -9.017, -2.192, -7.384, 7.952]},
            [18],
            {18: [1, -0.3917, -0.2747, -0.1852, -0.08768, -0.06009]},
            {8: 2.4404,
             17: 2.0057,
             18: 881.5305},
            9,
            2,
            categorical_idxs,
            continuous_idxs
        ))
    return equations
