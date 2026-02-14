"""
Submodel Base Classes
All submodels inherit from these base classes for consistency
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional


class SubModelBase(nn.Module):
    """Base class for all submodels"""

    def __init__(self):
        super().__init__()
        self.feature_name = None
        self.input_dim = None
        self.output_dim = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for training"""
        raise NotImplementedError

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform input data to submodel output (for meta-model)"""
        raise NotImplementedError

    def get_config(self) -> Dict:
        """Return model configuration"""
        return {
            "feature_name": self.feature_name,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
        }


class UnsupervisedSubModel(SubModelBase):
    """Base for unsupervised submodels (VAE, AE, etc.)"""

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space"""
        raise NotImplementedError

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to reconstruction"""
        raise NotImplementedError

    def compute_loss(self, x: torch.Tensor, reconstruction: torch.Tensor,
                    mu: Optional[torch.Tensor] = None,
                    logvar: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute reconstruction + regularization loss"""
        raise NotImplementedError


class SupervisedSubModel(SubModelBase):
    """Base for supervised submodels (regime classification, etc.)"""

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute supervised loss"""
        raise NotImplementedError


class HybridSubModel(SubModelBase):
    """Base for hybrid submodels (semi-supervised, multi-task)"""

    def compute_loss(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Compute multiple losses (supervised + unsupervised)"""
        raise NotImplementedError
