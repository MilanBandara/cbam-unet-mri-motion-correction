"""CBAM-U-Net for MRI motion artifact correction.

Reference implementation accompanying the paper:
    "Bridging the Gap: Supervised MRI Motion Artifact Correction via
     Data Synthesis: A CBAM-U-Net Approach"
    U.Y.G.M.K. Bandara, W.M.V.S. Herath, M. B. Dissanayake, S. C. Weerasinghe.

The package exposes the model architecture, paired data loaders, training
callbacks, evaluation metrics, visualisation helpers, and a k-space motion
synthesis sub-package.
"""

from cbam_unet.model import (
    AttentionBlock,
    Encoder_Block,
    Decoder_Block,
    Output_Block,
    artifact_removal_model,
)

__all__ = [
    "AttentionBlock",
    "Encoder_Block",
    "Decoder_Block",
    "Output_Block",
    "artifact_removal_model",
]

__version__ = "0.1.0"
