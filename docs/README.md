# CoPEM Framework Documentation

## Complete Documentation for Consensus-Driven Predictive Energy Management

Welcome to the CoPEM Framework documentation. This directory contains comprehensive technical documentation for all components of the framework.

---

## 📚 Available Documents

### 1. [MODELS_DOCUMENTATION.md](MODELS_DOCUMENTATION.md)

**Complete technical specification and source code documentation** for all AI models in the CoPEM framework:

- **Co-ESDRL Agent** (397,322 parameters)
  - SAC-based Deep Reinforcement Learning
  - Multi-objective brake blending optimization
  - Complete neural network architecture
  - Training procedures and hyperparameters

- **Eco-TES Transformer** (1,822,006 parameters)
  - Battery safe operating envelope prediction
  - Gated Temporal-Channel Attention (GTCA) mechanism
  - SOC-modulated positional encoding
  - Prediction performance metrics

- **Trust-Weighted Consensus**
  - Byzantine fault-tolerant state estimation
  - Dynamic trust score computation
  - Mahalanobis distance-based outlier detection
  - Performance under adversarial attacks

- **HOCBF Safety Filter**
  - High-Order Control Barrier Function formulation
  - Quadratic Programming solver
  - Real-time safety guarantees
  - Computational performance analysis

### 2. [ARCHITECTURE.md](ARCHITECTURE.md) *(Coming Soon)*

Detailed system architecture and component interactions.

### 3. [TRAINING.md](TRAINING.md) *(Coming Soon)*

Step-by-step training procedures for all models.

### 4. [API_REFERENCE.md](API_REFERENCE.md) *(Coming Soon)*

Complete API reference for the CoPEM framework.

---

## 🎯 Quick Navigation

### For Researchers
- **Understanding the Models**: Read [MODELS_DOCUMENTATION.md](MODELS_DOCUMENTATION.md)
- **Reproducing Results**: See main [README.md](../README.md) → "Reproducing Paper Results"
- **Citing Our Work**: See [README.md](../README.md) → "Citation"

### For Developers
- **API Usage**: See [../README.md](../README.md) → "Quick Start"
- **Installation**: See [../README.md](../README.md) → "Installation"
- **Contributing**: See [../CONTRIBUTING.md](../CONTRIBUTING.md)

### For Practitioners
- **Deployment Guide**: See [MODELS_DOCUMENTATION.md](MODELS_DOCUMENTATION.md) → Section 8
- **Hardware Requirements**: See [MODELS_DOCUMENTATION.md](MODELS_DOCUMENTATION.md) → Section 9.3
- **Performance Metrics**: See [MODELS_DOCUMENTATION.md](MODELS_DOCUMENTATION.md) → Section 3.6, 4.4, 5.5

---

## 📊 Key Performance Metrics

| Metric | Value | Validation |
|--------|-------|------------|
| **Energy Recovery Rate** | 36.5% (single-vehicle) | Euro NCAP tests |
| **Fleet Energy Improvement** | 187.9% | 6-vehicle cooperative |
| **Safety Rate** | 99.96% | 1000+ scenarios |
| **Fault Detection Rate** | 92.0% | 33% Byzantine attack |
| **Response Time** | 8.5 ms | Real-time system |
| **Total Parameters** | 2.2M | Deployable on edge |

---

## 🔬 Model Statistics

### Parameter Distribution

```
Total Parameters: 2,221,883
├── Eco-TES Transformer: 1,822,006 (82.0%)
│   ├── GTCA Blocks: 1,638,400 (73.7%)
│   ├── Envelope Predictors: 180,558 (8.1%)
│   └── Input Embedding: 2,048 (0.1%)
├── Co-ESDRL Agent: 397,322 (17.9%)
│   ├── Policy Network: 198,661 (8.9%)
│   └── Twin Q-Networks: 198,661 (8.9%)
├── Trust Consensus: 1,014 (0.0%)
└── HOCBF Controller: 1,541 (0.0%)
```

### Computational Performance

| Module | CPU Latency | GPU Latency | FLOPs |
|--------|-------------|-------------|-------|
| Consensus | 0.8 ms | 0.1 ms | 15.2K |
| Eco-TES | 12.5 ms | 1.2 ms | 2.8M |
| Co-ESDRL | 5.2 ms | 0.3 ms | 1.2M |
| HOCBF | 2.1 ms | 0.2 ms | 8.5K |
| **Total** | **20.6 ms** | **1.8 ms** | **4.0M** |

---

## 📖 Reading Guide

### For First-Time Users

1. Start with main [README.md](../README.md) for overview
2. Read [MODELS_DOCUMENTATION.md](MODELS_DOCUMENTATION.md) Section 1 (Architecture Overview)
3. Try "Quick Start" example in main README
4. Explore detailed model documentation as needed

### For Advanced Users

1. Review [MODELS_DOCUMENTATION.md](MODELS_DOCUMENTATION.md) Section 6-7 for training details
2. Check Section 8 for deployment optimization
3. See Section 9 for reproducibility specifications

### For Contributors

1. Read [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines
2. Review [MODELS_DOCUMENTATION.md](MODELS_DOCUMENTATION.md) for code structure
3. Check coding standards in main README

---

## 💾 Source Code

All model source code is available in:
```
copem/models/
├── co_esdrl_agent.py      # Co-ESDRL Agent implementation
├── eco_tes_transformer.py # Eco-TES Transformer implementation
└── __init__.py            # Module initialization
```

Complete API interface:
```
copem/api/
└── copem_api.py           # Main CoPEM API
```

---

## 🔗 External Resources

- **Paper**: "Nexus of Control: A Dynamic Consensus Framework for Energy-Positive Autonomous Emergency Braking"
- **GitHub**: https://github.com/PolyUDavid/CoPEM-Framework
- **Contact**: david.ko@connect.polyu.hk

---

## 📄 License

All documentation and code are released under the MIT License. See [LICENSE](../LICENSE) for details.

---

## ✉️ Contact

**Author**: DK  
**Institution**: Hong Kong Polytechnic University  
**Department**: Electrical and Electronic Engineering (EEE)  
**Email**: david.ko@connect.polyu.hk  
**GitHub**: https://github.com/PolyUDavid/CoPEM-Framework

For questions or suggestions about the documentation, please open an issue on GitHub or contact us directly.

---

**Last Updated**: December 17, 2025  
**Version**: 1.0.0

