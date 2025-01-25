import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer

class ECGFormerForClassificationOld(nn.Module):
    def __init__(self, model_dim, num_heads, num_encoder_layers, num_classes=2, dropout_rate=0.2):
        super(ECGFormerForClassification, self).__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        self.linear_proj = nn.Linear(2, model_dim)  
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=model_dim, 
                nhead=num_heads, 
                dropout=dropout_rate, 
                dim_feedforward=model_dim * 4, 
                norm_first=True,
                batch_first=True
            )
            for _ in range(num_encoder_layers)
        ])
        
        # Normalization layer
        self.norm = nn.LayerNorm(model_dim)
        
        # Dropout layer before classification
        self.dropout = nn.Dropout(dropout_rate)
        
        # Classification layer
        self.classifier = nn.Linear(model_dim, num_classes)

    def forward(self, src):
        # Project input to the required model dimension
        src = self.linear_proj(src)  # Now src is [batch_size, time_steps, model_dim]
        
        # Transformer encoder layers
        for encoder_layer in self.encoder_layers:
            residual = src
            src = encoder_layer(src)
            src = self.norm(src + residual)  # Check if it usefull

        # Apply dropout before classification
        src = self.dropout(src) # Check if it is usefull

        # Classification output (average pooling over time steps)
        classification_output = self.classifier(src.mean(dim=1))  # Pooling along the time dimension
        
        return classification_output
    
    import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer

class ECGFormerForClassification(nn.Module):
    def __init__(self, model_dim, num_heads, num_encoder_layers, num_classes=2, dropout_rate=0.2, input_length=None, use_learned_positional_encoding=True):
        """
        Parameters:
        - model_dim: Dimensionality of the Transformer model.
        - num_heads: Number of attention heads.
        - num_encoder_layers: Number of Transformer encoder layers.
        - num_classes: Number of output classes (default is 2).
        - dropout_rate: Dropout rate (default is 0.2).
        - input_length: Length of the ECG patch (mandatory for single-channel input).
        - use_learned_positional_encoding: Use learned positional encodings (default is True).
        """
        super(ECGFormerForClassification, self).__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        if input_length is None:
            raise ValueError("You must specify `input_length` for single-channel ECG data.")
        
        self.linear_proj = nn.Linear(input_length, model_dim)  # Adjust for single-channel input

        # Positional encoding
        if use_learned_positional_encoding:
            self.positional_encoding = nn.Parameter(torch.zeros(1, input_length, model_dim))  # Learned positional encoding
        else:
            self.register_buffer("positional_encoding", self.sinusoidal_encoding(input_length, model_dim))  # Sinusoidal positional encoding

        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dropout=dropout_rate,
                dim_feedforward=model_dim * 4,
                norm_first=True,
                batch_first=True
            )
            for _ in range(num_encoder_layers)
        ])

        # Normalization layer
        self.norm = nn.LayerNorm(model_dim)

        # Dropout layer before classification
        self.dropout = nn.Dropout(dropout_rate)

        # Classification layer
        self.classifier = nn.Linear(model_dim, num_classes)

    def sinusoidal_encoding(self, seq_len, embed_dim):
        """
        Generate sinusoidal positional encodings.
        """
        position = torch.arange(seq_len).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(seq_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, seq_len, embed_dim)

    def forward(self, src):
        # Project input to the required model dimension
        src = self.linear_proj(src)  # (batch_size, input_length, model_dim)
        print('[INFO] src.shape:', src.shape)
        position = self.positional_encoding 
        print('[INFO] position.shape:', position.shape)
        # Add positional encoding
        src = src + self.positional_encoding

        # Transformer encoder layers
        for encoder_layer in self.encoder_layers:
            residual = src
            src = encoder_layer(src)
            src = self.norm(src + residual)  # Optional residual connection

        # Apply dropout before classification
        src = self.dropout(src)  # Optional dropout
        classification_output = self.classifier(src.mean(dim=1))
        
        return classification_output
    
    def forward(self, src):
        # Reshape input: Add sequence length dimension if it's missing
        if len(src.shape) == 2:  # If shape is [batch_size, input_length]
            src = src.unsqueeze(1)  # Reshape to [batch_size, seq_len=1, input_length]

        # Project input to the required model dimension
        src = self.linear_proj(src)  # (batch_size, seq_len, model_dim)

        # Ensure positional encoding matches the input sequence length
        batch_size, seq_len, embed_dim = src.shape
        positional_encoding = self.positional_encoding[:, :seq_len, :]  # Truncate to match seq_len

        # Add positional encoding
        src = src + positional_encoding  # (batch_size, seq_len, embed_dim)

        # Transformer encoder layers
        for encoder_layer in self.encoder_layers:
            residual = src
            src = encoder_layer(src)
            src = self.norm(src + residual)  # Optional residual connection

        # Apply dropout before classification
        src = self.dropout(src)  # Optional dropout
        #Take the max
        classification_output = self.classifier(src.max(dim=1).values)

        return classification_output

class ECGFormerForSpecClassification(nn.Module):
    def __init__(self, model_dim, num_heads, num_encoder_layers, num_classes=2, dropout_rate=0.2, input_shape=None, use_learned_positional_encoding=True):
        """
        Parameters:
        - model_dim: Dimensionality of the Transformer model.
        - num_heads: Number of attention heads.
        - num_encoder_layers: Number of Transformer encoder layers.
        - num_classes: Number of output classes (default is 2).
        - dropout_rate: Dropout rate (default is 0.2).
        - input_shape: Shape of the spectrogram input (height, width).
        - use_learned_positional_encoding: Use learned positional encodings (default is True).
        """
        super(ECGFormerForSpecClassification, self).__init__()

        if input_shape is None or len(input_shape) != 2:
            raise ValueError("You must specify `input_shape` as (height, width) for spectrogram input.")

        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.input_height, self.input_width = input_shape

        # Linear projection to flatten the 2D spectrogram into a 1D sequence
        self.linear_proj = nn.Linear(self.input_width, model_dim)  # Projects width to model_dim

        # Positional encoding
        if use_learned_positional_encoding:
            self.positional_encoding = nn.Parameter(torch.zeros(1, self.input_height, model_dim))  # Learned positional encoding for height
        else:
            self.register_buffer("positional_encoding", self.sinusoidal_encoding(self.input_height, model_dim))

        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dropout=dropout_rate,
                dim_feedforward=model_dim * 4,
                norm_first=True,
                batch_first=True
            )
            for _ in range(num_encoder_layers)
        ])

        # Normalization layer
        self.norm = nn.LayerNorm(model_dim)

        # Dropout layer before classification
        self.dropout = nn.Dropout(dropout_rate)

        # Classification layer
        self.classifier = nn.Linear(model_dim, num_classes)

    def sinusoidal_encoding(self, seq_len, embed_dim):
        """
        Generate sinusoidal positional encodings.
        """
        position = torch.arange(seq_len).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(seq_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, seq_len, embed_dim)

    def forward(self, src):
        """
        Forward pass for the spectrogram classifier.
        Parameters:
        - src: Input spectrogram of shape (batch_size, height, width).
        """
        # Project the spectrogram along the width dimension
        src = self.linear_proj(src)  # (batch_size, height, model_dim)

        # Ensure positional encoding matches the input height
        batch_size, seq_len, embed_dim = src.shape
        positional_encoding = self.positional_encoding[:, :seq_len, :]  # Truncate to match seq_len

        # Add positional encoding
        src = src + positional_encoding  # (batch_size, height, model_dim)

        # Transformer encoder layers
        for encoder_layer in self.encoder_layers:
            residual = src
            src = encoder_layer(src)
            src = self.norm(src + residual)  # Optional residual connection

        # Apply dropout before classification
        src = self.dropout(src)  # Optional dropout

        # Global max pooling across the sequence dimension
        classification_output = self.classifier(src.max(dim=1).values)

        return classification_output
    