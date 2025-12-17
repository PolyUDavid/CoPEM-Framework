"""
CoPEM Framework - Consensus-Driven Predictive Energy Management
Official Implementation

This package provides the complete CoPEM framework for energy-efficient
autonomous emergency braking in electric vehicles.

Core Components:
- Co-ESDRL: Cooperative Energy-Saving Deep Reinforcement Learning Agent
- Eco-TES: Eco-Technical-Energy-Saving Transformer
- Trust-Weighted Consensus: Byzantine fault-tolerant state estimation
- HOCBF: High-Order Control Barrier Function safety filter

Authors: DK
Institution: Hong Kong Polytechnic University, Department of Electrical and Electronic Engineering
Date: December 15, 2025
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "DK"
__date__ = "December 15, 2025"

from copem.api.copem_api import (
    CoPEM_API,
    VehicleState,
    AEBScenario,
    EnergyRecoveryResult,
    BrakingMode,
    EnergySystemState
)

from copem.models.co_esdrl_agent import CoESDRLAgent
from copem.models.eco_tes_transformer import EcoTESTransformer

__all__ = [
    "CoPEM_API",
    "CoESDRLAgent",
    "EcoTESTransformer",
    "VehicleState",
    "AEBScenario",
    "EnergyRecoveryResult",
    "BrakingMode",
    "EnergySystemState",
]

# Framework information
FRAMEWORK_INFO = {
    "name": "CoPEM Framework",
    "full_name": "Consensus-Driven Predictive Energy Management",
    "version": __version__,
    "release_date": __date__,
    "paper_title": "Nexus of Control: A Dynamic Consensus Framework for Energy-Positive Autonomous Emergency Braking",
    "github": "https://github.com/PolyUDavid/CoPEM-Framework",
    "license": "MIT"
}

def print_info():
    """Print CoPEM framework information"""
    print("=" * 70)
    print(f"  {FRAMEWORK_INFO['name']} v{FRAMEWORK_INFO['version']}")
    print(f"  {FRAMEWORK_INFO['full_name']}")
    print("=" * 70)
    print(f"  Paper: {FRAMEWORK_INFO['paper_title']}")
    print(f"  Release Date: {FRAMEWORK_INFO['release_date']}")
    print(f"  GitHub: {FRAMEWORK_INFO['github']}")
    print(f"  License: {FRAMEWORK_INFO['license']}")
    print("=" * 70)
    print("\n  Key Achievements:")
    print("  ✅ 36.5% Energy Recovery (single-vehicle)")
    print("  ✅ 187.9% Fleet-Level Improvement")
    print("  ✅ 99.96% Safety Rate")
    print("  ✅ 92% Fault Detection (33% Byzantine attack)")
    print("  ✅ 8.5ms Response Time")
    print("=" * 70)

