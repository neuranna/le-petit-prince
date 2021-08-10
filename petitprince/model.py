
import torch.nn as nn
import torch

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
    self.encoder = nn.Linear(input_size, hidden_size)

    allowed_models = ['gru', 'lstm', 'rnn']
    assert model in allowed_models
    if self.model == "gru":
      self.rnn = nn.GRU(hidden_size, hidden_size, n_layers)
    elif self.model == "lstm":
      self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers)
    elif self.model == "rnn":
      self.rnn = nn.RNN(hidden_size, hidden_size, n_layers)

    self.decoder = nn.Linear(hidden_size, output_size)

  def forward(self, input, hidden):
    # batch_size = input.size(0)
    encoded = self.encoder(input.reshape(1, input.shape[0]))
    output, hidden = self.rnn(encoded.reshape(1, encoded.shape[0], -1), hidden)
    output = self.decoder(output.reshape(hidden.shape[0], -1))
    return output, hidden

  def init_hidden(self, batch_size):
    if self.model == "lstm":
      return (torch.zeros(self.n_layers, batch_size, self.hidden_size), torch.zeros(self.n_layers, batch_size, self.hidden_size))
    return torch.zeros(self.n_layers, batch_size, self.hidden_size)
