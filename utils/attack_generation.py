import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import to_complex
import matplotlib.pyplot as plt

def generate_fdia(conf):
    attack_base = conf["data"]["attack_base"]
    n_buses = conf["data"]["n_buses"]
    network_parameter_error = conf["attack"]["network_parameter_error"]
    state_variable_error = conf["attack"]["state_variable_error"]
    attack_bias = conf["attack"]["bias"]
    base_states_pd = pd.read_csv(attack_base)
    limits = pd.read_csv(conf["data"]["limits"])
    n_gen = (limits["PMAX"] != 0).sum()
    base_states = base_states_pd.to_numpy()
    mva_base = conf["data"]["mva_base"]

    attacked_busses = []
    new_states = []
    for i in tqdm(range(len(base_states))):
        # Add the non attack state
        new_states.append(base_states[i])

        true_state = np.copy(base_states[i][2:])

        # estamate the generation 
        p_deltas = []
        q_deltas = []
        for bus in range(n_buses):
            delta = (np.random.random() * 2 - 1) / n_gen
            p_shift = limits.loc[bus, "PMAX"] - limits.loc[bus, "PMIN"]
            q_shift = limits.loc[bus, "QMAX"] - limits.loc[bus, "QMIN"]
            p_deltas.append(p_shift * delta)
            q_deltas.append(q_shift * delta)
        p_deltas[0] = -sum(p_deltas[1:])
        q_deltas[0] = -sum(q_deltas[1:])

        for bus in range(n_buses):
            base_q = true_state[8 * bus + 1]
            true_state[8 * bus] = base_q + q_deltas[bus]
            base_p = true_state[8 * bus + 3]
            true_state[8 * bus + 2] = base_p + p_deltas[bus]

        # Go to per unit
        for bus in range(n_buses):
            bus_base_idx = 8 * bus
            for i in range(6):
                true_state[bus_base_idx + i] /= mva_base
        
        attack_state = np.copy(true_state)

        # apply network parameter errors
        ybus_base = 8 * n_buses
        for pos in range(ybus_base, ybus_base + n_buses * n_buses):
            true_value = to_complex(true_state[pos])
            true_state[pos] = true_value
            lower = true_value.real - network_parameter_error * true_value.real
            upper = true_value.real + network_parameter_error * true_value.real
            bias_real_value = lower + (upper - lower) * np.random.random()
            
            lower = true_value.imag - network_parameter_error * true_value.imag
            upper = true_value.imag + network_parameter_error * true_value.imag
            bias_imag_value = lower + (upper - lower) * np.random.random()

            attack_state[pos] = complex(bias_real_value, bias_imag_value)
        
        # Apply state variable errors
        for bus in range(n_buses):
            # voltage angle
            true_value = true_state[8 * bus + 6]
            lower = true_value - state_variable_error * true_value
            upper = true_value + state_variable_error * true_value
            bias_value = lower + (upper - lower) * np.random.random()
            attack_state[8 * bus + 6] = bias_value

            # voltage magnitude
            true_value = true_state[8 * bus + 7]
            lower = true_value - state_variable_error * true_value
            upper = true_value + state_variable_error * true_value
            bias_value = lower + (upper - lower) * np.random.random()
            attack_state[8 * bus + 7] = bias_value

        # Compute c to add to state and add it
        false_attack_state = np.copy(attack_state)
        nonzero_angles = set()
        nonzero_mags = set()
        for i in range(n_buses):
            if attack_state[8 * i + 6] != 0:
                nonzero_angles.add(i)
            if attack_state[8 * i + 7] != 0:
                nonzero_mags.add(i)
        
        available_buses = nonzero_mags.intersection(nonzero_angles)
    
        attacked_bus = random.sample(list(available_buses), 1)[0]
        attacked_busses.append(attacked_bus)
        bus_base_idx = 8 * attacked_bus
        true_angle = attack_state[bus_base_idx + 6]
        amount = random.uniform(0, 0.5) if random.random() < 0.5 else 1 - random.uniform(0, 0.5)
        attacked_angle = true_angle * ((1 - attack_bias) + 2 * attack_bias 
                                        * amount)
        assert true_angle != attacked_angle
        false_attack_state[bus_base_idx + 6] = attacked_angle

        true_magnitude = attack_state[bus_base_idx + 7]
        attacked_magnitude = true_magnitude * ((1 - attack_bias) + 2 * attack_bias
                                               * amount)
        assert true_magnitude != attacked_magnitude
        false_attack_state[bus_base_idx + 7] = attacked_magnitude

        # compute h(x\hat) and h(x\hhat + c)
        for bus_k in range(n_buses):
            # real power
            # h(x\hat)
            total_power = 0
            ybus_base = 8 * n_buses + n_buses * bus_k
            k_base_idx = 8 * bus_k
            theta_k = attack_state[k_base_idx + 6]
            v_k = attack_state[k_base_idx + 7]
            for bus_j in range(n_buses):
                j_base_idx = 8 * bus_j
                theta_j = attack_state[j_base_idx + 6]
                v_j = attack_state[j_base_idx + 7]
                radians = (np.pi / 180) * (theta_k - theta_j)
                admittance = attack_state[ybus_base + bus_j]
                bus_power = v_j * (admittance.real * np.cos(radians)
                                   + admittance.imag * np.sin(radians))
                total_power += bus_power
            total_power *= v_k
            gen_mw = attack_state[k_base_idx + 2]
            # Load = gen - bus_power
            attack_state[k_base_idx + 5] = gen_mw - total_power

            #h(x\hat + c)
            total_power = 0
            theta_k = false_attack_state[k_base_idx + 6]
            v_k = false_attack_state[k_base_idx + 7]
            for bus_j in range(n_buses):
                j_base_idx = 8 * bus_j
                theta_j = false_attack_state[j_base_idx + 6]
                v_j = false_attack_state[j_base_idx + 7]
                radians = (np.pi / 180) * (theta_k - theta_j)
                admittance = false_attack_state[ybus_base + bus_j]
                bus_power = v_j * (admittance.real * np.cos(radians) 
                                   + admittance.imag * np.sin(radians))
                total_power += bus_power
            total_power *= v_k
            gen_mw = false_attack_state[k_base_idx + 2]
            false_attack_state[k_base_idx + 5] = gen_mw - total_power

            # reactive power
            # h(x\hat)
            total_power = 0
            theta_k = attack_state[k_base_idx + 6]
            v_k = attack_state[k_base_idx + 7]
            for bus_j in range(n_buses):
                j_base_idx = 8 * bus_j
                theta_j = attack_state[j_base_idx + 6]
                v_j = attack_state[j_base_idx + 7]
                radians = (np.pi / 180) * (theta_k - theta_j)
                admittance = attack_state[ybus_base + bus_j]
                bus_power = v_j * (admittance.real * np.sin(radians)
                                   - admittance.imag * np.cos(radians))
                total_power += bus_power
            total_power *= v_k
            gen_mvar = attack_state[k_base_idx]
            attack_state[k_base_idx + 4] = gen_mvar - total_power

            # h(x\hat + c)
            total_power = 0
            theta_k = false_attack_state[k_base_idx + 6]
            v_k = false_attack_state[k_base_idx + 7]
            for bus_j in range(n_buses):
                j_base_idx = 8 * bus_j
                theta_j = false_attack_state[j_base_idx + 6]
                v_j = false_attack_state[j_base_idx + 7]
                radians = (np.pi / 180) * (theta_k - theta_j)
                admittance = false_attack_state[ybus_base + bus_j]
                bus_power = v_j * (admittance.real * np.sin(radians)
                                   - admittance.imag * np.cos(radians))
                total_power += bus_power
            total_power *= v_k
            gen_mvar = false_attack_state[k_base_idx]
            false_attack_state[k_base_idx + 4] = gen_mvar - total_power

        # a = h(x\hat + c) - h(x\hat)
        attack = false_attack_state[:8 * n_buses] - attack_state[:8 * n_buses]
        # z_a = true_state + a
        state = np.copy(true_state)
        for bus in range(n_buses):
            # mvar 
            k_base_idx = 8 * bus
            state[k_base_idx + 4: k_base_idx + 5] += attack[k_base_idx + 4: k_base_idx + 5]
        
        state[-2] = "yes"
        attacked = []
        for bus in range(n_buses):
            # real
            if attack[8 * bus + 5] > 0:
                attacked.append(2 * bus + 1)

            # reactive
            if attack[8 * bus + 4] > 0:
                attacked.append(2 * bus)

        state[-1] = attacked
        # test(true_state, n_buses)
        # Convert units
        for bus in range(n_buses):
            bus_base_idx = 8 * bus
            for i in range(6):
                state[bus_base_idx + i] *= mva_base

        state = np.hstack((np.array(["Date", "Time"]), state))
        new_states.append(state)

    new_states = np.row_stack(new_states)

    attacks = pd.DataFrame(new_states, columns=base_states_pd.columns)
    attacks.to_csv(conf["data"]["attack"], index=False)

    ax = plt.subplot()
    ax.hist(attacked_busses)
    plt.tight_layout()
    plt.savefig("Attacked_busses.png")

def test(state, n_buses):
    state = np.copy(state)
    for bus in range(n_buses):
        bus_idx = 6 * bus
        gen_mvar = state[bus_idx]
        load_mvar = state[bus_idx + 2]
        state[bus_idx] = gen_mvar - load_mvar

        gen_mw = state[bus_idx + 1]
        load_mw = state[bus_idx + 3]
        state[bus_idx + 1] = gen_mw - load_mw

    for bus in reversed(range(n_buses)):
        bus_idx = 6 * bus
        state = np.delete(state, bus_idx + 3, axis=0)
        state = np.delete(state, bus_idx + 2, axis=0)

    # Real power
    total_loss = 0
    for bus_k in range(n_buses):
        bus_loss = 0
        ybus_base = 4 * n_buses + bus_k * n_buses
        k_base_idx = 4 * bus_k
        power_k = state[k_base_idx + 1]
        theta_k = state[k_base_idx + 2]
        v_k = state[k_base_idx + 3]
        for bus_j in range(n_buses):
            j_base_idx = 4 * bus_j
            theta_j = state[j_base_idx + 2]
            v_j = state[j_base_idx + 3]
            radians = (np.pi / 180) * (theta_k - theta_j)
            admittance = state[ybus_base + bus_j]
            bus_power = v_j * (admittance.real * np.cos(radians)
                               + admittance.imag * np.sin(radians))
            bus_loss += bus_power
        bus_loss *= v_k
        bus_loss -= power_k
        total_loss += np.abs(bus_loss)
    average_real_power_loss = total_loss / n_buses
    print("Average real loss:", average_real_power_loss)

    # Reactive power
    total_loss = 0
    for bus_k in range(n_buses):
        bus_loss = 0
        ybus_base = 4 * n_buses + bus_k * n_buses
        k_base_idx = 4 * bus_k
        power_k = state[k_base_idx]
        theta_k = state[k_base_idx + 2]
        v_k = state[k_base_idx + 3]
        for bus_j in range(n_buses):
            j_base_idx = 4 * bus_j
            theta_j = state[j_base_idx + 2]
            v_j = state[j_base_idx + 3]
            radians = (np.pi / 180) * (theta_k - theta_j)
            admittance = state[ybus_base + bus_j]
            bus_power = v_j * (admittance.real * np.sin(radians)
                               - admittance.imag * np.cos(radians))
            bus_loss += bus_power
        bus_loss *= v_k
        bus_loss -= power_k
        total_loss += np.abs(bus_loss)
    average_reactive_power_loss = total_loss / n_buses
    print("Average reactive loss:", average_reactive_power_loss)
