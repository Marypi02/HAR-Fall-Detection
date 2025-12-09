import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pool_size=2, **_):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)

    def forward(self, x):
        """x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        return x"""

        # return self.pool(self.relu(self.conv(x)))
        return self.relu(self.bn(self.conv(x)))

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, **_):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose1d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride,
            padding=0
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        """x = self.deconv(x)
        x = self.relu(x)
        return x"""

        return self.relu(self.deconv(x))

class FeedForwardBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeedForwardBlock, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x
    
class ConvAutoencoder(nn.Module):
    """
    L'Encoder (Conv + Pool) serve per estrarre le feature e ridurre la dimensionalità.
    Il Decoder (Deconv) serve per la ricostruzione (vedere la differenza tra l'input originale e l'output ricostruito.
    GOAL: minimizzare l'errore di ricostruzione per la fase di Pre-Training
    """
    def __init__(self, in_channels, encoded_channels, kernel_size=3, pool_size=2):
        super(ConvAutoencoder, self).__init__()

        """max_compression = encoded_channels // 2 # Usiamo la metà della larghezza

        "--- ENCODER ---"
        self.encoder = nn.Sequential(
            # LAYER 1: in_channels(9) --> out_channels(256)
            ConvBlock(
                in_channels=in_channels,
                out_channels=encoded_channels,
                kernel_size=kernel_size,
                pool_size=pool_size
            ),
            # LAYER 2: in_channels(256) --> out_channels(128), in 64 canali mi deve compremire le 128 feature ottenute dal primo layer
            ConvBlock(
                in_channels=encoded_channels,
                out_channels=max_compression,
                kernel_size=kernel_size,
                pool_size=pool_size
            )
        )

        "--- DECODER ---"
        # LAYER 3: in_channels(128) --> out_channels(256)
        self.decoder = nn.Sequential(
            DeconvBlock(
                in_channels=max_compression,
                out_channels=encoded_channels,
                kernel_size=pool_size, # stessa dimensione usata in fase di encoding, per eseguire un upsampling e ripristinare la dimensione di spazio/tempo persa nel pooling
                stride=pool_size
            ),
            # LAYER 4: in_channels(256) --> out_channels(9)
            DeconvBlock(
                in_channels=encoded_channels,
                out_channels=in_channels,
                kernel_size=pool_size, # stessa dimensione usata in fase di encoding, per eseguire un upsampling e ripristinare la dimensione di spazio/tempo persa nel pooling
                stride=pool_size
            )
        )"""

        if isinstance(encoded_channels, int):
            encoded_channels = [encoded_channels]

        # --- ENCODER progressivo ---
        encoder_layers = []
        current_l = in_channels # partiamo con 9

        for depth in encoded_channels:
            encoder_layers.append(
                ConvBlock(
                    in_channels=current_l,
                    out_channels=depth,
                    kernel_size=kernel_size,
                    pool_size=pool_size
                )
            )
            current_l = depth

        self.encoder = nn.Sequential(*encoder_layers)
        self.latent_dim = current_l # depth finale

        # --- DECODER progressivo ---
        decoder_layers = []
        reversed_layers = encoded_channels[::-1]
        current_l = reversed_layers[0] 

        for i in range(len(reversed_layers)):
            # determino l'ouput del layer corrente
            if i < len(reversed_layers) - 1:
                depth = reversed_layers[i+1] # quindi scendo al livello precedente
            else:
                depth = reversed_layers[i] # sono arrivata al layer che contiene le caratteristiche alte (tipo 64)

            decoder_layers.append(
                DeconvBlock(
                    in_channels=current_l,
                    out_channels=depth,
                    kernel_size= pool_size,
                    stride = pool_size
                )
            )
            current_l = depth
        
        self.decoder_for_lstm = nn.Sequential(*decoder_layers)
        self.features_dim = current_l # questa è il valore delle features che andrà al lstm, da usare per la ricostruzione

        # --- RICOSTRUZIONE DELL'ULTIMO LAYER necessario per il pre-training e il calcolo della recostrution loss
        # torniamo all'input di partenza, ossia 9
        self.reconstructed_features = nn.Conv1d(self.features_dim, in_channels, kernel_size=1)

    def forward(self, x, is_har=True):
        
        # 1. Comprimi
        latent = self.encoder(x)

        # 2. Espandi il tempo (deocoder_for_lstm)
        feautures = self.decoder_for_lstm(latent)

        if is_har:
            # Caso LSTM: vengono restituite le feature 'ricche' del penultimo layer del decoder (64 canali, 128 timestep)
            return feautures
        else:
            # Caso Pre-Training: per calcolare l'errore di ricostruzione, i canali vengono schiacciati a 9
            return self.reconstructed_features(feautures)
        