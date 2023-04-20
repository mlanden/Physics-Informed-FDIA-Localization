
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

    return equations
