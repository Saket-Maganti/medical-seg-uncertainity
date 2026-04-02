import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class MCDropoutUNet(nn.Module):
    def __init__(self,
                 encoder_name="resnet34",
                 encoder_weights="imagenet",
                 in_channels=3,
                 num_classes=1,
                 dropout_p=0.3):
        super().__init__()
        self.dropout_p = dropout_p
        self._mc_active = False
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
            decoder_use_batchnorm=True,
        )
        # Single clean dropout layer after decoder — no injection
        self.mc_dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, x):
        features = self.unet.encoder(x)
        decoder_out = self.unet.decoder(features)
        if self.training or getattr(self, "_mc_active", False):
            decoder_out = F.dropout2d(
                decoder_out,
                p=self.dropout_p,
                training=True,
            )
        return self.unet.segmentation_head(decoder_out)

    def enable_mc(self):
        self._mc_active = True
        self.mc_dropout.train(True)

    def disable_mc(self):
        self._mc_active = False
        self.mc_dropout.train(False)


class DeepEnsemble(nn.Module):
    def __init__(self, n_models=5, **kwargs):
        super().__init__()
        self.n_models = n_models
        self.models = nn.ModuleList([MCDropoutUNet(**kwargs) for _ in range(n_models)])

    def forward(self, x):
        all_probs = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                probs = torch.sigmoid(model(x))
                all_probs.append(probs)
        all_probs = torch.stack(all_probs)
        return {
            "mean": all_probs.mean(0),
            "variance": all_probs.var(0),
            "all_probs": all_probs,
        }
