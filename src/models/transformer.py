import torch.nn as nn
import torch

class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=64):
        super().__init__()

        # input projection
        self.fc_in = nn.Linear(input_size, d_model)

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # output projection (THIS WAS MISSING IN YOUR RUNTIME)
        self.fc_out = nn.Linear(d_model, input_size)

    def forward(self, x):
    
        assert x.dim() == 3, f"Expected 3D input, got {x.shape}"
        assert x.shape[-1] == self.fc_in.in_features, \
            f"Feature mismatch: got {x.shape[-1]}, expected {self.fc_in.in_features}"

        x = self.fc_in(x)
        x = self.transformer(x)
        x = x[:, -1, :]

        
        assert hasattr(self, "fc_out"), "fc_out missing in model"

        x = self.fc_out(x)
        return x