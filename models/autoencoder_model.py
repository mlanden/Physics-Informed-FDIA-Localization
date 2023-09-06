import xxlimited
import torch
import torch.nn as nn

from .ics_model import ICSModel

class AutoencoderModel(ICSModel):

    def __init__(self, conf, categorical_values):
        super().__init__(conf, categorical_values)

        self.layer_sizes = conf["model"]["layer_sizes"]

        self.create_layers(self.layer_sizes[0])

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(len(self.layer_sizes) - 1):
            self.encoder.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1]))
            self.encoder.append(self.activation)
            self.decoder.insert(0, self.activation)
            self.decoder.insert(0, nn.Linear(self.layer_sizes[i + 1], self.layer_sizes[i]))

    def forward(self, unscaled_seq, scaled_seq):
        x = self.embed(unscaled_seq, scaled_seq)
        x = x[:, -1, :]
        x = self.input_linear(x)
        x = self.activation(x)
        
        for layer in self.encoder:
            x = layer(x)
        
        for layer in self.decoder:
            x = layer(x)
        out = x
        
        return self.outputs(out)