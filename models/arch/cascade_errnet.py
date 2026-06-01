import torch
import torch.nn as nn

from .default import DRNet


class CascadeERRNet(nn.Module):
    """
    IBCLN-style Cascaded Reflection Removal

    Stage1:
        Input -> DRNet -> T1

    Stage2:
        [Input,T1] -> DRNet -> Residual2
        T2 = T1 + Residual2

    Stage3:
        [Input,T2] -> DRNet -> Residual3
        T3 = T2 + Residual3
        
    Train:
        return T1, T2, T3
        
    Test:
        use T3
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n_feats=256,
        n_resblocks=13
    ):
        super().__init__()

        ###################################
        # Stage1
        ###################################
        self.stage1 = DRNet(
            in_channels=in_channels,
            out_channels=out_channels,
            n_feats=n_feats,
            n_resblocks=n_resblocks,
            norm=None,
            res_scale=0.1,
            se_reduction=8,
            bottom_kernel_size=1,
            pyramid=True
        )

        ###################################
        # Stage2
        ###################################
        self.stage2 = DRNet(
            in_channels=in_channels + out_channels,
            out_channels=out_channels,
            n_feats=n_feats,
            n_resblocks=n_resblocks,
            norm=None,
            res_scale=0.1,
            se_reduction=8,
            bottom_kernel_size=1,
            pyramid=True
        )

        ###################################
        # Stage3
        ###################################
        self.stage3 = DRNet(
            in_channels=in_channels + out_channels,
            out_channels=out_channels,
            n_feats=n_feats,
            n_resblocks=n_resblocks,
            norm=None,
            res_scale=0.1,
            se_reduction=8,
            bottom_kernel_size=1,
            pyramid=True
        )

    def forward(self, x):

        ###################################
        # Stage1
        ###################################
        t1 = self.stage1(x)

        ###################################
        # Stage2
        ###################################
        stage2_input = torch.cat([x, t1], dim=1)

        residual2 = self.stage2(stage2_input)

        t2 = t1 + residual2

        ###################################
        # Stage3
        ###################################
        stage3_input = torch.cat([x, t2], dim=1)

        residual3 = self.stage3(stage3_input)

        t3 = t2 + residual3

        ########################################
        # Deep Supervision
        ########################################
        return t1, t2, t3