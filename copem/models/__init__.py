"""
CoPEM Models Module

This module contains all AI models for the CoPEM framework:
- Co-ESDRL Agent: SAC-based reinforcement learning for brake blending
- Eco-TES Transformer: Battery envelope prediction with GTCA mechanism
- Consensus Estimator: Trust-weighted Byzantine fault-tolerant consensus
- HOCBF Controller: Formal safety guarantees via control barrier functions

All models were manually developed and trained by our research team.
Date: December 15, 2025
"""

from copem.models.co_esdrl_agent import CoESDRLAgent, PolicyNetwork, QNetwork
from copem.models.eco_tes_transformer import EcoTESTransformer, GTCABlock

__all__ = [
    "CoESDRLAgent",
    "PolicyNetwork",
    "QNetwork",
    "EcoTESTransformer",
    "GTCABlock",
]

