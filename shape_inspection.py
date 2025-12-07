import torch
import os

encoded_weights = "convAE_preTrain_outputs/pretrain_encoder_weights.pth"

if os.path.exists(encoded_weights):
    print(f"Loading weights from pre-training execution from {encoded_weights}...")

    weights_dict = torch.load(encoded_weights)

    print("Struttura e shape dei pesi dell'encoder ---")

    for name, param in weights_dict.items():
        print(f"Layer: {name}, Shape: {param.shape}")

else:
    print(f"File not found: {encoded_weights}. Execute pre-training process.")