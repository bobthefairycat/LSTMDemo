import torch  # PyTorch


class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_state_dim, num_layers, output_dim) -> None:
        """

        """
        super().__init__()
        self.hidden_size = hidden_state_dim
        self.num_layers = num_layers

        self.model = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_state_dim,
            num_layers=num_layers,
            batch_first=True
        )
        # find a way to experiment with taking info from all layers' hidden states
        # or just the last layer
        self.output_layer = torch.nn.Linear(hidden_state_dim, output_dim)

    def forward(self, x):
        lstm_output, _ = self.model(x)  # output is from the last layer of num_layers of lstm,
                                        # of size seq_len x hidden state size
        return self.output_layer(lstm_output)


