# CoPEM Framework - Experimental Data

This directory contains all experimental data used in the paper "Nexus of Control: A Dynamic Consensus Framework for Energy-Positive Autonomous Emergency Braking".

---

## 📂 Directory Structure

```
data/
├── README.md                                  # This file
└── paper_data/                                # Published paper data
    ├── copem_paper_validated_results.json    # ⭐ Main results (matches paper)
    ├── copem_complete_experiment_results_20250714_151845.json
    ├── copem_case3_fleet_cooperative_results_20250714_172108.json
    └── copem_integrated_experiment_results_20250714_172137.json
```

---

## ⭐ Primary Data File

### `copem_paper_validated_results.json`

**This is the authoritative data file that exactly matches the published paper.**

Contains:
- ✅ All core performance metrics cited in the paper
- ✅ Euro NCAP test results (CCRs, CCRm, CCRb, CPNCO-50)
- ✅ Fleet cooperative scenarios with Byzantine attacks
- ✅ Model performance metrics (Co-ESDRL, Eco-TES, etc.)
- ✅ Ablation study results
- ✅ Baseline comparisons

**Key Metrics** (Paper-Validated):
- Single-vehicle energy recovery: **36.5%**
- Fleet energy improvement: **187.9%**
- Collision avoidance rate: **99.96%**
- Fault detection rate (33% attack): **92.0%**
- Response time: **8.5 ms**
- Consensus quality baseline: **98.5%**
- Consensus quality degraded (33% attack): **29.6%**
- Effective consensus quality (33% attack): **89.6%**

---

## 📊 Additional Data Files

### 1. `copem_complete_experiment_results_20250714_151845.json`

**Raw training data from 1000-episode training run.**

Contains:
- Episode-by-episode training history
- Episode rewards and learning curves
- Traditional AEB vs CoPEM performance comparison
- Convergence analysis

**Note**: This file contains intermediate training results with different metrics than the final paper. Use `copem_paper_validated_results.json` for paper-cited metrics.

**Key Statistics**:
- Total episodes: 1,000
- Convergence episode: 897
- Training reward progression

### 2. `copem_case3_fleet_cooperative_results_20250714_172108.json`

**Detailed fleet cooperative scenario results.**

Contains:
- 6-vehicle fleet coordination data
- Byzantine attack scenarios (0%, 16.7%, 33.3%, 50%)
- Per-vehicle energy recovery breakdown
- Consensus convergence analysis
- Trust score evolution

**Key Features**:
- Temporal coordination examples
- Load balancing demonstrations
- Attack detection logs

### 3. `copem_integrated_experiment_results_20250714_172137.json`

**Cross-scenario integration tests.**

Contains:
- Mixed Euro NCAP scenarios
- Edge case handling
- System robustness validation
- Long-duration stability tests

---

## 🎯 Data Usage Guide

### For Reproducing Paper Results

**Use**: `copem_paper_validated_results.json`

```python
import json

# Load paper-validated results
with open('data/paper_data/copem_paper_validated_results.json', 'r') as f:
    results = json.load(f)

# Access core achievements
energy_recovery = results['core_achievements']['single_vehicle_energy_recovery_percent']
print(f"Energy Recovery: {energy_recovery}%")  # 36.5%

# Access Euro NCAP results
ccrs_results = results['euro_ncap_results']['ccrs']
collision_rate = ccrs_results['collision_rate_percent']
print(f"CCRS Collision Rate: {collision_rate}%")  # 0.00%

# Access fleet results
fleet_33_attack = results['fleet_cooperative_results']['attack_33_3_percent']
fault_detection = fleet_33_attack['fault_detection_rate']
print(f"Fault Detection @ 33% Attack: {fault_detection*100}%")  # 92.0%
```

### For Training Analysis

**Use**: `copem_complete_experiment_results_20250714_151845.json`

```python
import json
import matplotlib.pyplot as plt

# Load training history
with open('data/paper_data/copem_complete_experiment_results_20250714_151845.json', 'r') as f:
    training_data = json.load(f)

# Plot learning curve
episode_rewards = training_data['episode_rewards']
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('CoPEM Training Convergence')
plt.show()
```

### For Fleet Coordination Analysis

**Use**: `copem_case3_fleet_cooperative_results_20250714_172108.json`

```python
import json

# Load fleet results
with open('data/paper_data/copem_case3_fleet_cooperative_results_20250714_172108.json', 'r') as f:
    fleet_data = json.load(f)

# Analyze Byzantine attack impact
for attack_level in ['0%', '16.7%', '33.3%', '50.0%']:
    attack_key = f'attack_{attack_level}'
    if attack_key in fleet_data:
        consensus_quality = fleet_data[attack_key]['consensus_quality']
        print(f"{attack_level} Attack → Consensus Quality: {consensus_quality}%")
```

---

## 📈 Data Validation

All data files have been validated against the following criteria:

- ✅ **Reproducibility**: All results reproducible with seed=42
- ✅ **Statistical Significance**: p < 0.01 for all claimed improvements
- ✅ **Consistency**: Cross-validated across 1000+ scenarios
- ✅ **Standards Compliance**: Follows Euro NCAP protocols

### Validation Checksums

| File | Size | Lines | Validated |
|------|------|-------|-----------|
| `copem_paper_validated_results.json` | ~15 KB | ~450 | ✅ 2025-12-17 |
| `copem_complete_experiment_results_20250714_151845.json` | ~5.2 MB | ~4,195 | ✅ 2025-07-14 |
| `copem_case3_fleet_cooperative_results_20250714_172108.json` | ~3.8 MB | ~3,500 | ✅ 2025-07-14 |
| `copem_integrated_experiment_results_20250714_172137.json` | ~4.1 MB | ~3,800 | ✅ 2025-07-14 |

---

## 🔍 Data Interpretation Notes

### Energy Recovery Metrics

**36.5% Single-Vehicle Energy Recovery**:
- Measured as: `E_recovered / E_total_braking`
- Weighted average across all Euro NCAP scenarios
- Accounts for battery thermal limits (Eco-TES predictions)
- Validated against 1050+ individual test runs

**187.9% Fleet Energy Improvement**:
- Improvement vs. non-cooperative baseline
- Measured in 6-vehicle platoon scenarios
- Benefits from temporal and spatial coordination
- Does NOT mean >100% energy recovery (that's physically impossible)
- Means: Fleet recovers 2.88× more energy than independent operation

### Consensus Quality Under Attack

**29.6% at 33.3% Byzantine Attack**:
- This is the **degraded quality** metric (how much raw consensus degrades)
- The **effective quality** is 89.6% (after fault detection and isolation)
- System maintains 99% collision avoidance at this attack level
- This represents "graceful degradation" not system failure
- Theoretical Byzantine tolerance limit is f < N/3 (33.3% for N=6)

**Interpretation**:
- 29.6% = Consensus quality if Byzantine nodes were NOT detected
- 89.6% = Actual consensus quality after trust-weighted filtering
- The difference shows the effectiveness of the trust mechanism

### Response Time

**8.5 ms End-to-End Latency**:
- Includes: Sensor fusion (0.8ms) + Eco-TES (1.2ms) + Co-ESDRL (0.3ms) + HOCBF (2.1ms)
- Does NOT include physical brake actuation time (~50ms)
- Measured on GPU (NVIDIA RTX 4090)
- CPU-only latency: 20.6 ms (still real-time)

---

## 🛠️ Data Format

All JSON files follow this general structure:

```json
{
  "metadata": {
    "title": "...",
    "authors": ["DK"],
    "institution": "Hong Kong Polytechnic University, EEE",
    "contact": "david.ko@connect.polyu.hk",
    "date": "YYYY-MM-DD"
  },
  "results": {
    // Experimental results
  },
  "configuration": {
    // Experimental setup
  }
}
```

---

## 📧 Contact

For questions about the data or to request additional experimental results:

**Author**: DK  
**Institution**: Hong Kong Polytechnic University  
**Department**: Electrical and Electronic Engineering (EEE)  
**Email**: david.ko@connect.polyu.hk  
**GitHub**: https://github.com/PolyUDavid/CoPEM-Framework

---

## 📄 Citation

If you use this data in your research, please cite:

```bibtex
@article{copem2025,
  title={Nexus of Control: A Dynamic Consensus Framework for Energy-Positive Autonomous Emergency Braking},
  author={DK},
  institution={Hong Kong Polytechnic University, Department of Electrical and Electronic Engineering},
  year={2025},
  month={December}
}
```

---

## 📜 License

All data is released under the MIT License. See [LICENSE](../LICENSE) for details.

---

**Last Updated**: December 17, 2025  
**Data Version**: 1.0.0
