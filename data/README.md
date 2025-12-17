# CoPEM Experimental Data

This directory contains all experimental data from the paper:

> **"Nexus of Control: A Dynamic Consensus Framework for Energy-Positive Autonomous Emergency Braking"**

## Directory Structure

```
data/
├── paper_data/                    # Published experimental results
│   ├── copem_complete_experiment_results_20250714_151845.json
│   ├── copem_case3_fleet_cooperative_results_20250714_172108.json
│   └── copem_integrated_experiment_results_20250714_172137.json
│
└── training_datasets/             # Training data (not included in repo)
    └── README.md                  # Instructions for generating training data
```

## Paper Data Files

### 1. `copem_complete_experiment_results_20250714_151845.json`

**Complete experimental results** including:
- **Total Episodes**: 1,000 training episodes
- **Energy Recovery Efficiency**: 59.68% (normalized)
- **Safety Performance Index**: 98.47%
- **Consensus Quality**: 94.68%
- **Collision Avoidance Rate**: 99.7%
- **Reaction Time**: 80 ms
- **Inference Time**: 4.1 ms
- **Convergence Episode**: 897

**Comparison with Traditional AEB**:
| Metric | Traditional AEB | CoPEM | Improvement |
|--------|----------------|-------|-------------|
| Energy Recovery | 0.0% | 59.68% | ∞ |
| Reaction Time | 150 ms | 80 ms | 46.7% faster |
| Safety Performance | 85% | 98.47% | 15.8% |
| Collision Avoidance | 85% | 99.7% | 17.3% |

### 2. `copem_case3_fleet_cooperative_results_20250714_172108.json`

**Fleet cooperative scenario results** (6-vehicle platoon):
- **Fleet Size**: 6 vehicles
- **Byzantine Attack Scenarios**: 0%, 16.7%, 33.3%, 50%
- **Consensus Quality Under Attack**:
  - No attack: 98.5%
  - 33.3% attack: 89.6% (effective), 29.6% (degraded)
- **Energy Recovery**: 45.2% (no attack) → 41.2% (33% attack)
- **Fault Detection Rate**: 92.0% at 33% attack intensity

**Key Finding**: System maintains 99% collision avoidance even under theoretical Byzantine limit (f < N/3).

### 3. `copem_integrated_experiment_results_20250714_172137.json`

**Integrated cross-scenario analysis**:
- CCRs (Car-to-Car Rear Stationary)
- CCRm (Car-to-Car Rear Moving)
- CCRb (Car-to-Car Rear Braking)
- CPNCO-50 (Pedestrian Obstructed)

## Data Format

All JSON files follow this structure:

```json
{
  "timestamp": "2025-07-14T15:18:45",
  "total_episodes": 1000,
  "energy_recovery_efficiency": 0.5968,
  "safety_performance_index": 0.9847,
  "consensus_quality": 0.9468,
  "collision_avoidance_rate": 0.997,
  "reaction_time_ms": 80,
  "inference_time_ms": 4.1,
  "convergence_episode": 897,
  "episode_rewards": [...],
  "energy_recovery_history": [...],
  "safety_scores": [...]
}
```

## Reproducing Results

To reproduce these results:

```bash
# Run complete experimental suite
python experiments/run_euro_ncap_tests.py --scenarios all --trials 400

# Run fleet cooperative tests
python experiments/run_fleet_tests.py --fleet_size 6 --byzantine_ratio 0.33 --trials 100

# Compare with paper data
python scripts/validate_paper_results.py --data_dir data/paper_data/
```

## Data Integrity

All experimental data was collected between July 14-15, 2025, using:
- **Hardware**: Intel Core i9-13900K, 2× NVIDIA RTX 4090, 128GB RAM
- **Software**: Python 3.10.12, PyTorch 2.0.1, CUDA 11.8
- **Random Seeds**: Fixed for reproducibility

**SHA-256 Checksums**:
```
copem_complete_experiment_results_20250714_151845.json:
  [checksum will be generated]

copem_case3_fleet_cooperative_results_20250714_172108.json:
  [checksum will be generated]

copem_integrated_experiment_results_20250714_172137.json:
  [checksum will be generated]
```

## Citation

If you use this data in your research, please cite:

```bibtex
@article{copem2025,
  title={Nexus of Control: A Dynamic Consensus Framework for Energy-Positive Autonomous Emergency Braking},
  author={[Your Names]},
  journal={[Journal Name]},
  year={2025},
  month={December}
}
```

## License

This data is released under the MIT License. See [LICENSE](../LICENSE) for details.

---

**Last Updated**: December 15, 2025  
**Contact**: [your.email@institution.edu]

