import pandas as pd
import numpy as np
from torch import ge

from utils import to_complex


def generate_fdia(conf):
    attack_base = conf["data"]["attack_base"]
    n_buses = conf["data"]["n_buses"]
    network_parameter_error = conf["attack"]["network_parameter_error"]
    state_variable_error = conf["attack"]["state_variable_error"]
    base_states_pd = pd.read_csv(attack_base)
    base_states = base_states_pd.to_numpy()

    new_states = np.array()
    for i in range(10):#len(base_states)):
        # Add the non attack state
        new_states = np.vstack((new_states, base_states[i]))

        true_state = base_states[i]
        attack_state = np.copy(base_states[i])

        # apply network parameter errors
        ybus_base = 6 * n_buses
        for pos in range(ybus_base, ybus_base + n_buses * n_buses):
            true_value = to_complex(true_state[pos])
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
            true_value = true_state[6 * bus + 4]
            lower = true_value - state_variable_error * true_value
            upper = true_value + state_variable_error * true_value
            bias_value = lower + (upper - lower) * np.random.random()
            attack_state[6 * bus + 4] = bias_value

            # voltage magnitude
            true_value = true_state[6 * bus + 5]
            lower = true_value - state_variable_error * true_value
            upper = true_value + state_variable_error * true_value
            bias_value = lower + (upper - lower) * np.random.random()
            attack_state[6 * bus + 5] = bias_value

        # Compute c to add to state and add it
        false_attack_state = attack_state
        # TODO: add false data
        attacked_bus = np.random.randint(n_buses)

        # compute h(x\hat) and h(x\hhat + c)
        for bus_k in range(n_buses):
            # real power
            # h(x\hat)
            total_power = 0
            admittance_base = 6 * n_buses + n_buses * bus_k
            k_base_idx = 6 * bus_k
            theta_k = attack_state[k_base_idx + 4]
            v_k = attack_state[k_base_idx + 5]
            for bus_j in range(n_buses):
                j_base_idx = 6 * bus_j
                theta_j = attack_state[j_base_idx + 4]
                v_j = attack_state[j_base_idx + 5]
                radians = (np.pi / 180) * (theta_k - theta_j)
                admittance = attack_state[admittance_base + bus_j]
                bus_power = v_j * (admittance.real * np.cos(radians)
                                   + admittance.imag * np.sin(radians))
                total_power += bus_power
            total_power *= v_k
            gen_mw = attack_state[k_base_idx + 1]
            attack_state[k_base_idx + 3] = gen_mw - total_power

            #h(x\hat + c)
            total_power = 0
            theta_k = false_attack_state[k_base_idx + 4]
            v_k = false_attack_state[k_base_idx + 5]
            for bus_j in range(n_buses):
                j_base_idx = 6 * bus_j
                theta_j = false_attack_state[j_base_idx + 4]
                v_j = false_attack_state[j_base_idx + 5]
                radians = (np.pi / 180) * (theta_k - theta_j)
                admittance = false_attack_state[admittance_base + bus_j]
                bus_power = v_j * (admittance.real * np.cos(radians) 
                                   + admittance.imag * np.sin(radians))
                total_power += bus_power
            total_power *= v_k
            gen_mw = false_attack_state[k_base_idx + 2]
            false_attack_state[k_base_idx + 3] = gen_mw - total_power

            # reactive power
            # h(x\hat)
            total_power = 0
            theta_k = attack_state[k_base_idx + 4]
            v_k = attack_state[k_base_idx + 5]
            for bus_j in range(n_buses):
                j_base_idx = 6 * bus_j
                theta_j = attack_state[j_base_idx + 4]
                v_j = attack_state[j_base_idx + 5]
                radians = (np.pi / 180) * (theta_k - theta_j)
                admittance = attack_state[admittance_base + bus_j]
                bus_power = v_j * (admittance.real * np.sin(radians)
                                   - admittance.imag * np.cos(radians))
                total_power += bus_power
            total_power *= v_k
            gen_mvar = attack_state[k_base_idx]
            attack_state[k_base_idx + 2] = gen_mvar - total_power

            # h(x\hat + c)
            total_power = 0
            theta_k = false_attack_state[k_base_idx + 4]
            v_k = false_attack_state[k_base_idx + 5]
            for bus_j in range(n_buses):
                j_base_idx = 6 * bus_j
                theta_j = false_attack_state[j_base_idx + 4]
                v_j = false_attack_state[j_base_idx + 5]
                radians = (np.pi / 180) * (theta_k - theta_j)
                admittance = false_attack_state[admittance_base + bus_j]
                bus_power = v_j * (admittance.real * np.sin(radians)
                                   - admittance.imag * np.cos(radians))
                total_power += bus_power
            total_power *= v_k
            gen_mvar = false_attack_state[k_base_idx]
            false_attack_state[k_base_idx + 2] = gen_mvar - total_power

        # a = h(x\hat + c) - h(x\hat)
        attack = false_attack_state - attack_state

        # z_a = true_state + a
        state = np.copy(true_state)
        for bus in range(n_buses):
            # mvar 
            k_base_idx = 6 * bus
            state[k_base_idx + 2: k_base_idx + 3] += attack[k_base_idx + 2: k_base_idx + 3]
        
        state[-2] = "yes"
        state[-1] = attacked_bus
        test(state, n_buses)
        new_states = np.vstack((new_states, state))

    attacks = pd.DataFrame(new_states, columns=base_states_pd.columns)
    attacks.to_csv(conf["data"]["attack"], index=False)

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
        state = np.delete(state, bus_idx + 3, axis=1)
        state = np.delete(state, bus_idx + 2, axis=1)

    # Real power
    total_loss = 0
    for bus_k in range(n_buses):
        bus_loss = 0
        admittance_base = 4 * n_buses + bus_k * n_buses
        k_bus_base = 4 * bus_k
        power_k = state[k_bus_base + 1]
        theta_k = state[k_bus_base + 2]
        v_k = state[k_bus_base + 3]
        for bus_j in range(n_buses):
            j_bus_base = 4 * bus_j
            theta_j = state[j_bus_base + 2]
            v_j = state[j_bus_base + 3]
            radians = (np.pi / 180) * (theta_k - theta_j)
            admittance = state[admittance_base + bus_j]
            bus_power = v_j * (admittance.real * np.cos(radians)
                               + admittance.imag * np.sin(radians))
            bus_loss += bus_power
        bus_loss *= v_k
        bus_loss -= power_k
        total_loss += bus_loss
    average_real_power_loss = total_loss / n_buses
    print("Average real loss:", average_real_power_loss)

    total_loss = 0
    for bus_k in range(n_buses):
        buss_loss = 0
        admittance_base = 4 * n_buses + bus_k * n_buses
        k_bus_base = 4 * bus_k
        power_k = state[k_bus_base + 1]
        theta_k = state[k_bus_base + 2]
        v_k = state[k_bus_base + 3]
        for bus_j in range(n_buses):
            j_bus_base = 4 * bus_j
            theta_j = state[j_bus_base + 2]
            v_j = state[j_bus_base + 3]
            radians = (np.pi / 180) * (theta_k - theta_j)
            admittance = state[admittance_base + bus_j]
            bus_power = v_j * (admittance.real * np.sin(radians)
                               - admittance.imag * np.cos(radians))
            bus_loss += bus_power
        bus_loss *= v_k
        bus_loss -= power_k
        total_loss += bus_loss
    average_reactive_power_loss = total_loss / n_buses
    print("Average reactive loss:", average_reactive_power_loss)
            
