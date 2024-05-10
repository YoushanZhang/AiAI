import torch.nn as nn
import torch
import torch.nn.init as init

# Nonlinearity class for activation functions like Swish
class Swish(nn.Module):
    def __init__():
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

# Main residual block with 2 convolution layers and a skip connection
class ResConv1DBlock(nn.Module):
    def __init__(self, n_in, n_state, dilation=1, activation='silu', norm=None, dropout=None):
        super().__init__()

        # Padding for convolution with dilation
        padding = dilation
        
        # Add dropout
        self.dropout = dropout
        
        # Configure normalization
        self.norm = norm
        if norm == "LN":
            self.norm1 = nn.LayerNorm(n_in)
            self.norm2 = nn.LayerNorm(n_in)
        elif norm == "GN":
            self.norm1 = nn.GroupNorm(32, n_in)
            self.norm2 = nn.GroupNorm(32, n_in)
        elif norm == "BN":
            self.norm1 = nn.BatchNorm1d(n_in)
            self.norm2 = nn.BatchNorm1d(n_in)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        # Configure activation
        if activation == "relu":
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()
        elif activation == "silu":
            self.activation1 = Swish()
            self.activation2 = Swish()
        elif activation == "gelu":
            self.activation1 = nn.GELU()
            self.activation2 = nn.GELU()
        else:
            raise ValueError("Unsupported activation type")

        # Convolution layers with skip connection
        self.conv1 = nn.Conv1d(n_in, n_state, 3, padding=padding, dilation=dilation)
        self.conv_skip = nn.Conv1d(n_state, n_state, 1, stride=1, padding=0)
        self.conv2 = nn.Conv1d(n_state, n_in, 1, padding=0)

        # Dropout layer if specified
        if self.dropout:
            self.drop = nn.Dropout(dropout)

        # Initialize weights with suitable initialization
        init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        init.kaiming_normal_(self.conv_skip.weight, nonlinearity='relu')
        init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')

    def forward(self, x):
        x_orig = x

        # Apply normalization and activation
        if self.norm == "LN":
            x = self.norm1(x.transpose(-2, -1)).transpose(-2, -1)
        else:
            x = self.norm1(x)

        x = self.activation1(x)

        # First convolution
        x = self.conv1(x)

        # Dropout after first convolution if needed
        if self.dropout:
            x = self.drop(x)

        # Apply skip connection between two convolution layers
        skip = self.conv_skip(x)
        
        # Normalization and activation again
        if self.norm == "LN":
            skip = self.norm2(skip.transpose(-2, -1)).transpose(-2, -1)
        else:
            skip = self.norm2(skip)

        skip = self.activation2(skip)

        # Apply the second convolution
        x = self.conv2(skip)

        # Final skip connection with the original input
        x = x + x_orig

        return x


# Main ResNet1D class
class Resnet1D(nn.Module):
    def __init__(self, n_in, n_depth, dilation_growth_rate=1, reverse_dilation=True, activation='relu', norm=None, dropout=None):
        super().__init__()
        
        # Create residual blocks with the specified configuration
        blocks = [ResConv1DBlock(n_in, n_in, dilation=dilation_growth_rate ** depth, activation=activation, norm=norm, dropout=dropout) for depth in range(n_depth)]
        
        if reverse_dilation:
            blocks = blocks[::-1]
        
        self.model = nn.Sequential(*blocks)

    def forward(self, x):        
        return self.model(x)
