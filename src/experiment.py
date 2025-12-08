import hydra
import torch
from omegaconf import DictConfig # Sono gli oggetti di configurazione di Hydra, consentono di accedere ai parametri definiti nei file YAML. 
import lightning as lit
from net.base import Net
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, BaseFinetuning
from dataset.loaders import load_har, load_wisdm

import lightning.pytorch.callbacks as cb 

from omegaconf import OmegaConf

import os

class unfreezeConvAE(BaseFinetuning):
    def __init__(self, unfreeze_at_epoch: int = 5):
        super().__init__()
        self.unfreeze_at_epoch = unfreeze_at_epoch

    def freeze_before_training(self, pl_module):
        # Chiamata automaticamente prima del training
        # congeliamo il convAE
        print("[Callback] ConvAE weights are frozen initially.")
        self.freeze(pl_module.conv_ae)

    def finetune_function(self, pl_module, current_epoch, optimizer):
        # chiamata ad ogni epoch
        if current_epoch == self.unfreeze_at_epoch:
            print(f"[Callback] Unfreezing ConvAE weights at epoch {current_epoch}!")

            self.unfreeze_and_add_param_group(
                modules=pl_module.conv_ae,
                optimizer=optimizer,
                # train_bn=True # se nel ConvBlock ho una BatchNorm1d
            )

@hydra.main(config_path="../cfg", config_name="base", version_base=None)
def main(cfg: DictConfig):
    lit.seed_everything(cfg.seed) 
    wandb_logger = WandbLogger(**cfg.wandb) # wandb login 
    model = Net(cfg.net)

    # --- Caricamento dei pesi ottenuti durante la fase di pre-training
    convAE_weights_path = "convAE_preTrain_outputs/convAE_weights.pth"

    if os.path.exists(convAE_weights_path):
        print("Loading ConvAE weights for HAR Fine-Tuning...")

        weights_dict = torch.load(convAE_weights_path)

        try:
            model.conv_ae.load_state_dict(weights_dict)
            print("Encoder weights loaded successfully! Fine-Tuning start...")

            # --- Provo a congelare i pesi dell'encoder così che non vengano influenzati dal lstm
            """for param in model.conv_ae.parameters():
                param.requires_grad = False 
            print("Encoder weights are NOW FROZEN.")
            model.conv_ae.eval()"""

        except Exception as e:
            print(f"Error loading weights dict: {e}")
    else:
        print("Pre-Trained weights not found!")

    train_loader, val_loader, test_loader = load_har(**cfg.dataset)

    checkpoint = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filename="best_HAR-{epoch:02d}-{val_acc:.4f}"
    )

    unfreeze_weights = unfreezeConvAE(unfreeze_at_epoch=10)

    trainer = lit.Trainer(logger=wandb_logger, callbacks=[
        cb.EarlyStopping(
            monitor="val_acc",
            patience=15,
            verbose=True,
            mode="max", 
            min_delta=1e-3
        ),
        checkpoint,
        unfreeze_weights
    ], **cfg.trainer)

    print('Training HAR model...')
    trainer.fit(model, train_loader, val_loader)
    
    print('Testing...')

    try:
        # Ottiene il percorso del modello migliore salvato dal callback
        best_model_path = trainer.checkpoint_callback.best_model_path
        # Carica il modello migliore tramite quel percorso
        best_model = Net.load_from_checkpoint(best_model_path, cfg=cfg.net)
        # Test sul modello migliore
        trainer.test(best_model, test_loader)
    except AttributeError:
        # Se il test fallisce o non è stato salvato un checkpoint, esegue il testing sull'ultimo stato
        print("Could not load best checkpoint. Testing last model state.")
        trainer.test(model, test_loader)
    
    hyperparams_dict = OmegaConf.to_container(cfg, resolve=True)
    hyperparams_dict["info"] = {  
        "num_params": get_num_params(model),
    }
    wandb_logger.log_hyperparams(hyperparams_dict)  

def get_num_params(module):
    """
    Returns the number of parameters in a Lightning module.
    
    Args:
        module (lightning.pytorch.LightningModule): The Lightning module to get the number of parameters for.
    
    Returns:
        int: The number of parameters in the module.
    """
    total_params = sum(p.numel() for p in module.parameters() )
    return total_params




if __name__ == "__main__":
    os.environ['HYDRA_FULL_ERROR'] = "1"
    main()