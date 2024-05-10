import torch.nn as nn
import torch
import torch.nn.init as init

# Swish nonlinearity
class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

# Improved residual block with three convolutions, dropout, and normalization
class ResConv1DBlock(nn.Module):
    def __init__(self, n_in, n_state, dilation=1, activation='silu', norm=None, dropout=None):
        super().__init__()

        self.dropout = dropout
        self.norm = norm
        
        # Select normalization
        def get_norm(n):
            if norm == "LN":
                return nn.LayerNorm(n)
            elif norm == "GN":
                return nn.GroupNorm(32, n)
            elif norm == "BN":
                return nn.BatchNorm1d(n)
            else:
                return nn.Identity()
        
        self.norm1 = get_norm(n_in)
        self.norm2 = get_norm(n_state)
        self.norm3 = get_norm(n_in)

        # Select activation
        def get_activation(a):
            if a == "relu":
                return nn.ReLU()
            elif a == "silu":
                return Swish()
            elif a == "gelu":
                return nn.GELU()
            elif a == "leaky_relu":
                return nn.LeakyReLU(0.01)
            else:
                raise ValueError("Unsupported activation type")

        self.activation1 = get_activation(activation)
        self.activation2 = get_activation(activation)
        self.activation3 = get_activation(activation)

        # Convolution layers with dropout and normalization
        self.conv1 = nn.Conv1d(n_in, n_state, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(n_state, n_state, 3, padding=dilation, dilation=dilation)
        self.conv3 = nn.Conv1d(n_state, n_in, 1)  # Back to input dimensions

        if dropout:
            self.drop = nn.Dropout(dropout)

        # Initialize weights
        init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')

    def forward(self, x):
        x_orig = x

        # Normalize and activate
        x = self.norm1(x)
        x = self.activation1(x)

        # First convolution
        x = self.conv1(x)
        
        # Apply dropout if specified
        if self.dropout:
            x = self.drop(x)

        # Normalize and activate again
        x = self.norm2(x)
        x = self.activation2(x)

        # Second convolution
        x = self.conv2(x)

        # Normalize, activate, and apply the final convolution
        x = self.norm3(x)
        x = self.activation3(x)
        x = self.conv3(x)

        # Apply skip connection
        x = x + x_orig

        return x


# ResNet1D with multiple residual blocks
class Resnet1D(nn.Module):
    def __init__(self, n_in, n_depth, dilation_growth_rate=1, reverse_dilation=True, activation='relu', norm=None, dropout=None):
        super().__init__()
        
        # Create residual blocks
        blocks = [ResConv1DBlock(n_in, n_in, dilation=dilation_growth_rate ** depth, activation=activation, norm=norm, dropout=dropout) for depth in range(n_depth)]
        
        if reverse_dilation:
            blocks = blocks[::-1]  # Reverse the order if needed
        
        self.model = nn.Sequential(*blocks)

    def forward(self, x):        
        return self.model(x)
