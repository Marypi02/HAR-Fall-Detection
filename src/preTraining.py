import hydra
import os
import torch
from omegaconf import DictConfig 
import lightning as lit
from net.base import Net
import torch.nn.functional as F
import torch.optim as optim
from hydra.utils import instantiate
from lightning.pytorch.loggers import WandbLogger
from dataset.loaders import load_har, load_wisdm
from net.cnn import ConvAutoencoder
import lightning.pytorch.callbacks as cb 

# --- FASE 1: Pre-Training non supervisionato per addestrare i pesi dell'encoder riudencendo l'errore di ricostruzione ---
class ConvAE_pretraining(lit.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.conv_ae = instantiate(cfg.embed)
        self.cfg = cfg

    def forward(self, x):
        return self.conv_ae(x, is_har=False) # con is_har=False, viene restituito l'ultimo layer del decoder
    
    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_reconstructed = self(x)

        # Loss Reconstruction Error: Mean Squared Error tra l'output e l'input di partenza
        loss = F.mse_loss(x_reconstructed, x)
        self.log("train_loss_convAE", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_reconstructed = self(x)
        loss = F.mse_loss(x_reconstructed, x)
        self.log("val_loss_convAE", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.optimizer, self.parameters())

        # Lr scheduler per ridurre lr se la val_loss si blocca
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=0.5, 
            patience=5
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss_convAE",
                "interval": "epoch",
                "frequency": 1,
            },
        }
    
@hydra.main(config_path="../cfg", config_name="base", version_base=None)
def main(cfg: DictConfig):
    lit.seed_everything(cfg.seed)
    wandb_logger = WandbLogger(**cfg.wandb)
    model = ConvAE_pretraining(cfg.net)

    train_loader, val_loader, _ = load_har(**cfg.dataset)

    trainer = lit.Trainer(
        logger=wandb_logger, 
        callbacks=[ 
            cb.EarlyStopping(
                monitor="val_loss_convAE",
                patience=20,
                verbose=True,
                mode="min", 
                min_delta=1e-4
            ),
            cb.ModelCheckpoint(
                monitor="val_loss_convAE",
                mode="min",
                save_top_k=1,
                filename="{best_val_loss_convAE:.4f}"
            )
        ], 
        **cfg.trainer
    )

    print("Starting ConvAE Pre-Training...")
    trainer.fit(model, train_loader, val_loader)

    # --- Salvataggio completo dei pesi ---
    best_checkpoint_path = trainer.checkpoint_callback.best_model_path
    best_model = ConvAE_pretraining.load_from_checkpoint(best_checkpoint_path, cfg=cfg.net)

    if not os.path.exists("convAE_preTrain_outputs"):
        os.mkdir("convAE_preTrain_outputs")

    final_weights_path = "convAE_preTrain_outputs/convAE_weights.pth"
    torch.save(best_model.conv_ae.state_dict(), final_weights_path)

    print(f"\nEncoder weights saved for Transfer Learning at: {final_weights_path}")

if __name__ == "__main__":
    os.environ['HYDRA_FULL_ERROR'] = "1"
    main()