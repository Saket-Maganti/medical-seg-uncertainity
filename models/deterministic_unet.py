from models.unet_mc import MCDropoutUNet


class DeterministicUNet(MCDropoutUNet):
    def __init__(self, **kwargs):
        kwargs["dropout_p"] = 0.0
        super().__init__(**kwargs)
