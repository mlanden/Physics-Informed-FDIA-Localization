
from .arx_equation import ARXEquation


def build_equations(dataset, continuous_idxs):
    equations = []
    if dataset == "swat":
        equations.append(ARXEquation(
            [0, 1],
            {0: [3.73e-14, -3.9e-14, 2.15e-15, 3.498e-15],
             1: [1]},
            [1],
            {1: [1, -4.486e-18, -3.318e-16]},
            {0: 0.0111,
             1: 712.1462},
            2,

            continuous_idxs
        ))

    return equations
