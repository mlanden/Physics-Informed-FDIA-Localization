from .real_power_equation import RealPowerEquation
from .reactive_power_equation import ReactivePowerEquation
from .real_power_graph_equation import RealPowerGraphEquation
from .reactive_power_graph_equation import ReactivePowerGrapghEquation
from .power_graph_equation import PowerGraphEquation


def build_equations(conf, categorical_idxs=None, continuous_idxs=None):
    dataset = conf["data"]["type"]
    equations = []
    if dataset == "grid":
        n_buses = conf["data"]["n_buses"]
        graph = conf["model"]["graph"]
        # for i in range(n_buses):
        #     equations.append(RealPowerGraphEquation(n_buses, i))
        #     equations.append(ReactivePowerGrapghEquation(n_buses, i))
        if graph:
            equations.append(PowerGraphEquation(n_buses))
        else:
            for i in range(n_buses):
                equations.append(RealPowerEquation(n_buses, i))
                equations.append(ReactivePowerEquation(n_buses, i))

    return equations
