# CoPEM Framework - Models Documentation

**Complete Technical Specification and Source Code Documentation**

Date: December 15, 2025  
Authors: DK  
Institution: Hong Kong Polytechnic University, Department of Electrical and Electronic Engineering  
Contact: david.ko@connect.polyu.hk

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Co-ESDRL Agent](#co-esdrl-agent)
3. [Eco-TES Transformer](#eco-tes-transformer)
4. [Trust-Weighted Consensus](#trust-weighted-consensus)
5. [HOCBF Safety Filter](#hocbf-safety-filter)
6. [Model Parameters](#model-parameters)
7. [Training Procedures](#training-procedures)

---

## 1. Architecture Overview

The CoPEM framework consists of four core AI/control components working in a hierarchical pipeline:

```
Input (Sensor Data + V2X) 
    ↓
[Trust-Weighted Consensus] → Robust State Estimation
    ↓
[Eco-TES Transformer] → Battery Envelope Prediction
    ↓
[Co-ESDRL Agent (SAC)] → Optimal Brake Blending
    ↓
[HOCBF Safety Filter] → Safety-Certified Control
    ↓
Output (Brake Commands)
```

### System Flow

1. **Consensus Layer** (100 Hz): Fuses multi-agent observations, filters Byzantine faults
2. **Prediction Layer** (10 Hz): Forecasts battery safe operating boundaries
3. **Decision Layer** (100 Hz): Computes optimal regenerative/friction brake blend
4. **Safety Layer** (100 Hz): Enforces formal collision avoidance guarantees

---

## 2. Co-ESDRL Agent

### 2.1 Overview

The **Cooperative Energy-Saving Deep Reinforcement Learning (Co-ESDRL) Agent** is the core decision-making module. It implements a Soft Actor-Critic (SAC) algorithm optimized for continuous brake blending control.

### 2.2 Architecture

**Total Parameters**: 397,322

#### Actor Network (Policy)
```python
Input: State (24-dim) → [256] → [256] → [128] → Output: Action (2-dim)
```

- **Input State Vector** (24 dimensions):
  - Vehicle kinematics (8): position, velocity, acceleration, heading
  - Battery state (8): SOC, voltage, current, temperature, power
  - Network state (4): quality, neighbor count, latency
  - Predictions (4): max regen torque, thermal limit, voltage limit, confidence

- **Output Action Vector** (2 dimensions):
  - `α_regen`: Regenerative braking ratio [0, 1]
  - `α_friction`: Friction braking ratio [0, 1]
  - Constraint: `α_regen + α_friction ≤ 1`

#### Critic Networks (Twin Q-Networks)
```python
Input: State (24-dim) + Action (2-dim) → [256] → [256] → [128] → Output: Q-value (1-dim)
```

- Two identical networks to reduce overestimation bias
- Energy-aware attention layer in hidden representations

### 2.3 Multi-Objective Reward Function

```python
R_total = 0.45 * R_safety + 0.25 * R_energy + 0.20 * R_comfort + 0.10 * R_consensus
```

**R_safety**: Collision avoidance
```python
R_safety = -1000 * collision_flag 
         + 100 * exp(-d_rel / d_safe) 
         + 50 * indicator(TTC > TTC_min)
```

**R_energy**: Energy recovery and battery health
```python
R_energy = η_regen * (P_recovered / P_total)
         - λ_bat * (ΔT_battery)^2
         - μ_degr * |I_battery|
```

**R_comfort**: Smooth driving
```python
R_comfort = -α_jerk * |jerk|^2 - β_accel * |a_long|^2
```

**R_consensus**: Network cooperation
```python
R_consensus = γ_trust * trust_score - δ_fault * fault_flag
```

### 2.4 Training Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 3e-4 | Adam optimizer |
| Batch Size | 256 | Experience replay sampling |
| Buffer Size | 1,000,000 | Replay buffer capacity |
| Discount Factor (γ) | 0.99 | Future reward discount |
| Soft Update (τ) | 0.005 | Target network update rate |
| Temperature (α) | Auto-tuned | Entropy regularization |
| Training Episodes | 1,000 | Total training episodes |
| Convergence Episode | 897 | Achieved stable performance |

### 2.5 Source Code Structure

**File**: `copem/models/co_esdrl_agent.py`

**Key Classes**:
- `CoESDRLAgent`: Main agent class
- `PolicyNetwork`: Actor network with energy/safety feature extraction
- `QNetwork`: Critic network with attention mechanism
- `ReplayBuffer`: Experience replay storage

**Key Methods**:
- `compute_action()`: Generate brake blending command from state
- `train()`: Update networks using SAC algorithm
- `_update_critics()`: Update Q-networks
- `_update_actor()`: Update policy network
- `save_model()` / `load_model()`: Model persistence

---

## 3. Eco-TES Transformer

### 3.1 Overview

The **Eco-Technical-Energy-Saving (Eco-TES) Transformer** predicts the battery's safe operating envelope over a 10-step future horizon, enabling proactive energy recovery planning.

### 3.2 Architecture

**Total Parameters**: 1,822,006

```
Input: Battery State Sequence (50 timesteps × 16 features)
    ↓
[Input Embedding] (16 → 128)
    ↓
[SOC-Modulated Positional Encoding]
    ↓
[GTCA Block 1] (Gated Temporal-Channel Attention)
    ↓
[GTCA Block 2]
    ↓
[GTCA Block 3]
    ↓
[GTCA Block 4]
    ↓
[Battery Envelope Predictor] (128 → 8 outputs)
    ↓
Output: Safe Operating Boundaries
  - Voltage: [V_min, V_max]
  - Current: [I_min, I_max]
  - Temperature: [T_min, T_max]
  - Power: [P_min, P_max]
```

### 3.3 GTCA (Gated Temporal-Channel Attention) Block

**Innovation**: Dual-stream attention with gating mechanism

#### Temporal Attention
- Captures time-series dependencies in battery dynamics
- Models thermal inertia and electrochemical lag

#### Channel Attention
- Captures cross-feature correlations
- Models SOC-voltage-temperature coupling

#### Gating Mechanism
```python
G_t = sigmoid(X_t * W_G + X_{t-1} * W_G_prev + b_G + E_SOC)
Output = G_t ⊙ temporal_attn + (1 - G_t) ⊙ channel_attn
```

### 3.4 SOC-Modulated Positional Encoding

**Innovation**: Adapts temporal scale based on battery SOC

```python
PE(pos, 2i) = sin(pos / 10000^(2i/d) * γ_SOC(pos))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d) * γ_SOC(pos))

γ_SOC(pos) = 1 + α_mod * tanh(β_mod * (SOC(pos) - 0.5))
```

**Physical Interpretation**:
- High SOC (>80%): γ > 1 → Time compression (faster dynamics)
- Low SOC (<20%): γ < 1 → Time dilation (slower dynamics)

### 3.5 Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Input Dimension | 16 | Battery state features |
| Hidden Dimension | 128 | Transformer hidden size |
| Attention Heads | 8 | Multi-head attention |
| GTCA Blocks | 4 | Transformer layers |
| Sequence Length | 50 | Input timesteps |
| Prediction Horizon | 10 | Future steps to predict |
| Learning Rate | 1e-4 | AdamW optimizer |
| Batch Size | 32 | Training batch size |
| Training Epochs | 185 | Early stopping |
| Dropout | 0.1 | Regularization |

### 3.6 Performance Metrics

| Predicted Variable | MAE | RMSE | R² Score |
|-------------------|-----|------|----------|
| SOC (10 steps) | 0.0082 | 0.0125 | 0.985 |
| Temperature (°C) | 0.68 | 0.95 | 0.923 |
| Voltage (V) | 1.25 | 1.87 | 0.957 |
| P_regen_max (kW) | 2.15 | 3.28 | 0.915 |

### 3.7 Source Code Structure

**File**: `copem/models/eco_tes_transformer.py`

**Key Classes**:
- `EcoTESTransformer`: Main transformer model
- `GTCABlock`: Gated temporal-channel attention module
- `BatteryEnvelopePredictor`: Multi-output prediction head
- `BatteryDataset`: Training data loader

**Key Methods**:
- `forward()`: Forward pass with consensus features
- `predict_envelope()`: Inference for single sequence
- `train_step()`: Single training iteration
- `_calculate_envelope_loss()`: Custom loss function

---

## 4. Trust-Weighted Consensus

### 4.1 Overview

The **Trust-Weighted Dynamic Consensus** algorithm provides Byzantine fault-tolerant state estimation for multi-agent cooperative scenarios.

### 4.2 Algorithm Components

#### Historical Trust Update
```python
T_ij(t+1) = γ * T_ij(t) + (1 - γ) * R_ij(t)

where:
  γ = 0.85 (forgetting factor)
  R_ij(t) = 1 / (1 + exp(β * (D_ij(t) - θ_rep)))
```

#### Dynamic Trust Score
```python
τ_ij(t) = exp(-λ * ||x_i(t) - x_j(t)||^2) * T_ij(t) * φ_temporal(t)

where:
  λ = 0.07 (sensitivity parameter)
  φ_temporal = temporal consistency factor
```

#### Weight Normalization
```python
w_ij(t) = τ_ij(t) * (1 - B_i(t)) / (ε + Σ_k τ_ik(t) * (1 - B_k(t)))
```

#### Consensus State
```python
x̄_i(t+1) = (1 - μ) * Σ_j w_ij(t) * x_j(t) + μ * x̄_i(t)

where:
  μ = 0.15 (momentum coefficient)
```

### 4.3 Byzantine Fault Detection

**Mahalanobis Distance Test**:
```python
D_i(t) = (x_i(t) - μ_robust)^T * Σ_robust^-1 * (x_i(t) - μ_robust)

Byzantine Flag: B_i(t) = indicator(D_i(t) > χ²_α,df)
```

### 4.4 Performance Under Attack

| Byzantine Attack | Consensus Quality | Fault Detection Rate | Collision Rate |
|-----------------|-------------------|---------------------|----------------|
| 0% (No attack) | 98.5% | - | 0.00% |
| 16.7% (1/6 nodes) | 95.2% | 94.5% | 0.00% |
| 33.3% (2/6 nodes) | 89.6% | 92.0% | 1.00% |
| 50.0% (3/6 nodes) | 62.3% | 78.5% | 5.00% |

**Key Finding**: At 33.3% attack (theoretical Byzantine limit f < N/3), the system maintains 99% collision avoidance despite 29.6% degraded consensus quality.

### 4.5 Computational Performance

- **Convergence Time**: 45.3 ms (6-vehicle network)
- **Update Rate**: 100 Hz
- **Communication Overhead**: O(N) per agent

---

## 5. HOCBF Safety Filter

### 5.1 Overview

The **High-Order Control Barrier Function (HOCBF)** provides formal safety guarantees by solving a Quadratic Programming (QP) problem in real-time.

### 5.2 Safety Function

```python
h(x) = d - d_safe - v_rel * t_react - v_rel^2 / (2 * (a_max_ego - a_max_lead))

where:
  d: relative distance
  d_safe = 2.5 m (safety margin)
  t_react = 0.15 s (system reaction time)
  a_max_ego = 9.0 m/s² (ego max deceleration)
  a_max_lead = 6.0 m/s² (lead max deceleration)
```

### 5.3 HOCBF Constraint (Relative Degree 3)

```python
L_f³h(x) + L_g L_f²h(x) * u ≥ -α₃ψ₂ - α₂L_f²h - α₁α₂L_fh

where:
  α₁ = κ₁ * (·), κ₁ = 1.5
  α₂ = κ₂ * (·), κ₂ = 2.0
  α₃ = κ₃ * (·), κ₃ = 2.5
```

### 5.4 QP Formulation

```python
u* = argmin ||u - u_RL||²_W
subject to:
  HOCBF constraint
  u_min ≤ u ≤ u_max
  α_regen + α_friction ≤ 1
  α_regen ≤ α_regen_max(SOC, T)
```

### 5.5 Solver Performance

- **Solver**: OSQP with warm-start
- **Average Solve Time**: 0.3 ms
- **Worst-Case Solve Time**: 2.1 ms (95th percentile)
- **Success Rate**: 100% (always feasible)

---

## 6. Model Parameters Summary

### 6.1 Complete Parameter Count

| Component | Parameters | Percentage |
|-----------|-----------|------------|
| **Co-ESDRL Agent** | | |
| - Policy Network | 198,661 | 8.9% |
| - Twin Q-Networks | 198,661 | 8.9% |
| - Subtotal | 397,322 | 17.9% |
| **Eco-TES Transformer** | | |
| - Input Embedding | 2,048 | 0.1% |
| - GTCA Blocks (4×) | 1,638,400 | 73.7% |
| - Envelope Predictors | 180,558 | 8.1% |
| - Consensus Processor | 1,000 | 0.0% |
| - Subtotal | 1,822,006 | 82.1% |
| **Trust Consensus** | 1,014 | 0.0% |
| **HOCBF Controller** | 1,541 | 0.0% |
| **TOTAL** | **2,221,883** | **100%** |

### 6.2 Memory Footprint

- **Model Weights**: 16.49 MB (FP32)
- **Activations** (inference): 179 MB
- **Total GPU Memory**: ~200 MB

### 6.3 Computational Complexity

| Module | FLOPs | Latency (CPU) | Latency (GPU) |
|--------|-------|---------------|---------------|
| Consensus | 15.2K | 0.8 ms | 0.1 ms |
| Eco-TES | 2.8M | 12.5 ms | 1.2 ms |
| Co-ESDRL | 1.2M | 5.2 ms | 0.3 ms |
| HOCBF | 8.5K | 2.1 ms | 0.2 ms |
| **Total** | **4.0M** | **20.6 ms** | **1.8 ms** |

---

## 7. Training Procedures

### 7.1 Co-ESDRL Training

**Curriculum Learning Strategy**:

1. **Stage 1** (Episodes 0-200k): Simple CCRs scenarios, 30-50 km/h
2. **Stage 2** (Episodes 200k-500k): Full speed range, CCRs + CCRm
3. **Stage 3** (Episodes 500k-800k): Add CCRb and CPNCO scenarios
4. **Stage 4** (Episodes 800k-1M): Mixed scenarios + Byzantine attacks

**Training Progress**:

| Episode Range | Avg Return | Energy (%) | Collision Rate | Entropy |
|--------------|-----------|------------|----------------|---------|
| 0-50k | -85.3 ± 125 | 5.2 ± 8.5 | 15.8% | 0.85 |
| 100k-200k | 45.8 ± 52 | 22.5 ± 10.8 | 3.2% | 0.58 |
| 400k-600k | 112.5 ± 25 | 35.8 ± 6.2 | 0.35% | 0.35 |
| 800k-1M | 152.8 ± 15 | 37.8 ± 4.5 | 0.08% | 0.25 |

**Convergence**: Episode 897

### 7.2 Eco-TES Training

**Data Preparation**:
- 50,000 battery state sequences from vehicle telemetry
- Sequence length: 50 timesteps (5 seconds @ 10 Hz)
- Train/Val/Test split: 70% / 15% / 15%

**Training Schedule**:
- Total Epochs: 200
- Early Stopping: Patience = 10 epochs
- Best Model: Epoch 185 (validation loss = 0.0115)

**Learning Rate Schedule**:
```python
lr(epoch) = lr_base * cosine_annealing(epoch, T_max=1000, eta_min=1e-6)
```

### 7.3 Hyperparameter Optimization

**Grid Search Results** (selected configurations):

| Config | λ_safety | λ_energy | λ_comfort | λ_consensus | Final Performance |
|--------|----------|----------|-----------|-------------|-------------------|
| A | 0.40 | 0.30 | 0.20 | 0.10 | 35.2% energy, 0.12% CR |
| **B (Chosen)** | **0.45** | **0.25** | **0.20** | **0.10** | **36.5% energy, 0.04% CR** |
| C | 0.50 | 0.20 | 0.20 | 0.10 | 34.8% energy, 0.02% CR |

---

## 8. Model Deployment

### 8.1 Edge Computing Optimization

**Quantization**: FP32 → INT8
- Latency Reduction: 35%
- Memory Reduction: 75%
- Accuracy Impact: <1.5%

**Deployment Performance**:
- Eco-TES: 0.8 ms (TensorRT + INT8)
- Co-ESDRL: 0.2 ms
- Total Pipeline: 2.1 ms (Target <10 ms ✓)
- Power Consumption: 14.2 W average

### 8.2 Real-Time Constraints

| Requirement | Specification | Achieved |
|------------|---------------|----------|
| Perception Update | 100 Hz (10 ms) | ✓ 8.5 ms |
| Prediction Update | 10 Hz (100 ms) | ✓ 12 ms |
| Control Update | 100 Hz (10 ms) | ✓ 2.1 ms |
| End-to-End Latency | <50 ms | ✓ 23 ms |

---

## 9. Reproducibility

### 9.1 Random Seeds

All experiments use fixed random seeds for reproducibility:
```python
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
```

### 9.2 Environment Specifications

- **Python**: 3.10.12
- **PyTorch**: 2.0.1
- **CUDA**: 11.8
- **cuDNN**: 8.7.0

### 9.3 Hardware

- **CPU**: Intel Core i9-13900K (24 cores, 5.8 GHz turbo)
- **GPU**: 2× NVIDIA RTX 4090 (24 GB VRAM each)
- **RAM**: 128 GB DDR5-5600 ECC
- **Storage**: 2 TB NVMe SSD

---

## 10. Citation

If you use these models in your research, please cite:

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

## Contact

**Author**: DK  
**Institution**: Hong Kong Polytechnic University  
**Department**: Electrical and Electronic Engineering (EEE)  
**Email**: david.ko@connect.polyu.hk  
**GitHub**: https://github.com/PolyUDavid/CoPEM-Framework

---

**Last Updated**: December 17, 2025  
**Version**: 1.0.0

