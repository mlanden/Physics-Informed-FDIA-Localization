from typing import List

import torch

from .equation import Equation


class ARXEquation(Equation):

    def __init__(self, input_idxs: list, input_coefficients: dict, output_idxs: list, output_coefficients: dict,
                 mean_values: dict, actuator_idx: int, target_state: int, categorical_idxs: dict, continuous_idxs: list):
        self.target_state = target_state
        self.actuator_idx = actuator_idx
        self.categorical_idx = list(categorical_idxs.keys()).index(actuator_idx)
        self.categorical_features = categorical_idxs
        self.continuous_idxs = continuous_idxs
        self.mean_values = mean_values
        self.output_coefficients = output_coefficients
        self.output_idxs = output_idxs
        self.input_coefficients = input_coefficients
        self.input_idxs = input_idxs

    def evaluate(self, states):
        if states[-1][self.actuator_idx] != self.target_state:
            return 0
        # print("in")
        input_total = self._compute_total(states, self.input_idxs, self.input_coefficients)
        # print("\nout")
        output_total = self._compute_total(states, self.output_idxs, self.output_coefficients)
        # for idx in self.output_idxs:
        #     output_total += states[-1, idx] - self.mean_values[idx]
        # quit()
        # check for actuator state
        target = self.target_state
        is_target = (states[-1, self.actuator_idx] == target).astype(float)
        # print(input_total, output_total, is_target)

        return abs(input_total - output_total) * is_target

    def confidence_loss(self, input_states: torch.Tensor, network_outputs: List[torch.Tensor]) -> torch.Tensor:
        input_total = self._compute_loss(input_states, self.input_idxs, self.input_coefficients)
        for idx in self.input_idxs:
            input_idx = self.continuous_idxs.index(idx)
            input_total += self.input_coefficients[idx][0] * (network_outputs[0][:, input_idx] - self.mean_values[idx]
                                                               + input_states[:, -1, idx] 
                                                              )

        output_total = self._compute_loss(input_states, self.output_idxs, self.output_coefficients)
        for idx in self.output_idxs:
            output_idx = self.continuous_idxs.index(idx)
            output_total += self.output_coefficients[idx][0] * (network_outputs[0][:, output_idx] - self.mean_values[idx]
                                                                 + input_states[:, -1, idx] 
                                                                )

        # check for actuator state
        target = self.target_state
        if self.categorical_features[self.actuator_idx] == 2:
            target -= 1
        actuator_states = torch.argmax(network_outputs[self.categorical_idx + 1], dim=1)
        is_target = (actuator_states == target).float()
        return (input_total - output_total) ** 2 * is_target

    def _compute_total(self, states, indexes, coefficients):
        total = 0
        for idx in indexes:
            for i, coefficient in enumerate(coefficients[idx]):
                # print(-i - 1, coefficient, end=", ")
                total += coefficient * (states[-i - 1, idx] - self.mean_values[idx])
            # print()
        return total

    def _compute_loss(self, input_states, indexes, coefficients, start=1):
        total = torch.zeros((input_states.shape[0]), device=input_states.device)
        for idx in indexes:
            for i, coefficient in enumerate(coefficients[idx][start:]):
                total += coefficient * (input_states[:, -i - 1, idx] - self.mean_values[idx])
        return total
