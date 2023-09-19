
from .arx_equation import ARXEquation
from .real_power_equation import RealPowerEquation


def build_equations(conf, categorical_idxs, continuous_idxs):
    dataset = conf["data"]["type"]
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
        equations.append(ARXEquation(
            [17, 27],
            {17: [-0.3779],
             27: [23.16]},
            [28],
            {28: [1, -0.9468, -0.0005742, -0.0001626, -0.004112, -0.00428, -0.04173]},
            {17: 1.8410,
             27: 1.7097,
             28: 885.043},
             30,
             2,
             categorical_idxs,
             continuous_idxs
        ))
        equations.append(ARXEquation(
            [40],
            {40: [-3.488]},
            [46],
            {46: [1, -1.263, 0.2569]},
            {40: 0.7384,
             46: 191.4908},
             42,
             2,
             categorical_idxs,
             continuous_idxs
        ))
    elif dataset == "grid":
        n_buses = conf["data"]["n_buses"]
        admittance = conf["data"]["ybus"]
        equations.append(RealPowerEquation(n_buses, admittance))
    return equations
