import torch
import torch.nn as nn


class PetitPrinceNet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size=None, model='rnn', n_layers=1):
        super().__init__()

        if not output_size:
            output_size = input_size

        self.model = model.lower()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        allowed_models = ['gru', 'lstm', 'rnn']
        assert self.model in allowed_models

        self.encoder = nn.Linear(input_size, hidden_size)
        self.rnn = eval(f'nn.{self.model}(hidden_size, hidden_size, n_layers)')
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        encoded = self.encoder(input.reshape(1, input.shape[0]))
        output, hidden = self.rnn(encoded.reshape(1, encoded.shape[0], -1), hidden)
        output = self.decoder(output.reshape(hidden.shape[0], -1))
        return output, hidden

    def init_hidden(self, batch_size):
        if self.model == "lstm":
            return (torch.zeros(self.n_layers, batch_size, self.hidden_size),
                    torch.zeros(self.n_layers, batch_size, self.hidden_size))
        return torch.zeros(self.n_layers, batch_size, self.hidden_size)
