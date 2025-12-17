# Mathematical Models for Simulation Implementation

## 1. Vehicle Dynamics (3-DOF Model)

### Equations of Motion

The vehicle state is represented by position (x, y), velocity (v_x, v_y), heading (θ), and yaw rate (ω_z).

**Longitudinal Dynamics**:
```
m(dv_x/dt - v_y·ω_z) = F_x,f·cos(δ) - F_y,f·sin(δ) + F_x,r - F_aero - F_grade - F_roll
```

**Lateral Dynamics**:
```
m(dv_y/dt + v_x·ω_z) = F_x,f·sin(δ) + F_y,f·cos(δ) + F_y,r
```

**Yaw Dynamics**:
```
I_z·dω_z/dt = l_f·(F_y,f·cos(δ) + F_x,f·sin(δ)) - l_r·F_y,r
```

Where:
- m = 1850 kg (vehicle mass)
- I_z = 3200 kg·m² (yaw inertia)
- δ = steering angle (rad)
- l_f = 1.3 m, l_r = 1.6 m (axle distances)

### Force Components

**Aerodynamic Drag**:
```
F_aero = 0.5 · ρ_air · C_d · A_front · v_x²
       = 0.5 · 1.225 · 0.28 · 2.3 · v_x²
       = 0.394 · v_x² [N]
```

**Rolling Resistance**:
```
F_roll = C_r · m · g · cos(θ_grade)
       = 0.01 · 1850 · 9.81 · cos(θ_grade)
       ≈ 181.5 [N]  (flat road)
```

**Grade Force**:
```
F_grade = m · g · sin(θ_grade)
```

### Tire Forces (Pacejka Magic Formula)

**Longitudinal Force**:
```
F_x = D_x · sin(C_x · arctan(B_x·κ - E_x·(B_x·κ - arctan(B_x·κ))))
```

Where:
- κ = (v_wheel - r·ω) / v_wheel (slip ratio)
- D_x = F_z · μ_peak (peak force)
- B_x = 10.0, C_x = 1.9, E_x = 0.97

**Lateral Force** (simplified):
```
F_y = D_y · sin(C_y · arctan(B_y·α))
```

Where α is the slip angle.

---

## 2. Battery Model (Second-Order Thévenin)

### State-of-Charge (SOC)

**Coulomb Counting**:
```
dSOC/dt = -I_batt / (3600 · Q_nominal)
```

Where:
- I_batt = battery current (A, negative for discharge)
- Q_nominal = 214.3 Ah (75 kWh / 350 V)

### Voltage Dynamics

**Terminal Voltage**:
```
V_terminal = V_OC(SOC) - I_batt · R_0(SOC, T) - V_RC
```

**RC Network**:
```
dV_RC/dt = -V_RC / (R_1·C_1) + I_batt / C_1
```

Where:
- R_0 = 0.05 Ω (at 25°C, 50% SOC)
- R_1 = 0.02 Ω, C_1 = 2000 F (typical RC parameters)

### Thermal Dynamics

**Battery Temperature**:
```
m_batt · c_p · dT_batt/dt = I_batt² · R_0 - h · A_surf · (T_batt - T_amb)
```

Expanding:
```
500 · 1000 · dT_batt/dt = I² · 0.05 - 10 · 4.0 · (T_batt - T_amb)
```

Simplified:
```
dT_batt/dt = 10^-7 · I² - 8×10^-5 · (T_batt - T_amb)
```

### Power Limits

**Maximum Regenerative Power**:
```
P_regen_max = min(P_motor_max, P_battery_max(SOC, T))
```

**Battery Power Limit**:
```
P_battery_max = V_terminal · I_max · η_regen
```

Where I_max depends on SOC and temperature:
```
I_max(SOC, T) = I_rated · f_SOC(SOC) · f_T(T)

f_SOC(SOC) = {
    0.5,  if SOC > 0.95
    0.75, if 0.8 < SOC ≤ 0.95
    1.0,  if 0.2 < SOC ≤ 0.8
    0.75, if 0.1 < SOC ≤ 0.2
    0.5,  if SOC ≤ 0.1
}

f_T(T) = {
    0.5,  if T < 0°C or T > 45°C
    0.8,  if 0°C ≤ T < 15°C or 40°C < T ≤ 45°C
    1.0,  if 15°C ≤ T ≤ 40°C
}
```

---

## 3. Brake Force Distribution

### Total Brake Force

**Desired Deceleration**:
```
F_brake_total = m · a_desired
```

Where a_desired comes from the CoPEM controller.

### Regenerative vs Friction Split

**Regenerative Component**:
```
F_regen = min(α_regen · F_brake_total, P_regen_max / v_x)
```

**Friction Component**:
```
F_friction = α_friction · F_brake_total
```

**Constraint**:
```
α_regen + α_friction ≤ 1
F_regen + F_friction = F_brake_total
```

### Front/Rear Distribution

For simplicity, we use fixed 70/30 split:
```
F_brake,front = 0.7 · (F_regen + F_friction)
F_brake,rear = 0.3 · (F_regen + F_friction)
```

---

## 4. Energy Recovery Calculation

### Energy During Braking Event

**Total Kinetic Energy**:
```
E_kinetic = 0.5 · m · (v_initial² - v_final²)
```

**Recovered Energy**:
```
E_recovered = ∫[t0→tf] P_regen(t) · dt
```

Discretized:
```
E_recovered = Σ[i=1→N] P_regen(t_i) · Δt
```

Where:
```
P_regen(t) = F_regen(t) · v_x(t) · η_regen
```

**Energy Recovery Efficiency**:
```
η_recovery = E_recovered / E_kinetic
```

### Battery SOC Change

```
ΔSOC = -∫[t0→tf] I_batt(t) dt / (3600 · Q_nominal)
     = E_recovered / (V_nominal · Q_nominal)
```

---

## 5. Safety Metrics

### Time-to-Collision (TTC)

```
TTC = d_rel / v_rel   (if v_rel > 0)
    = ∞              (if v_rel ≤ 0)
```

Where:
- d_rel = relative distance
- v_rel = v_ego - v_lead (relative velocity)

### Minimum Safe Distance

Based on constant deceleration assumption:
```
d_safe = v_rel · t_react + v_rel² / (2 · (a_ego_max - a_lead_max)) + d_margin
```

Where:
- t_react = 0.15 s (system reaction time)
- a_ego_max = 9.0 m/s² (maximum ego deceleration)
- a_lead_max = 6.0 m/s² (assumed lead deceleration)
- d_margin = 2.5 m (safety margin)

### Collision Detection

```
collision = {
    True,   if d_rel < d_collision and v_rel > 0
    False,  otherwise
}
```

Where d_collision = 1.0 m (vehicle overlap threshold).

---

## 6. Simulation Loop Structure

### Time Discretization

- **Control timestep**: Δt = 0.01 s (100 Hz)
- **Integration method**: 4th-order Runge-Kutta (RK4)

### RK4 Integration

For state x with dynamics dx/dt = f(x, u, t):

```
k1 = f(x_n, u_n, t_n)
k2 = f(x_n + 0.5·Δt·k1, u_n, t_n + 0.5·Δt)
k3 = f(x_n + 0.5·Δt·k2, u_n, t_n + 0.5·Δt)
k4 = f(x_n + Δt·k3, u_n, t_n + Δt)

x_{n+1} = x_n + (Δt/6)·(k1 + 2·k2 + 2·k3 + k4)
```

### State Vector

```
x = [p_x, p_y, v_x, v_y, θ, ω_z, SOC, T_batt]ᵀ
```

### Control Loop Sequence

1. **Sensor Measurement** (with noise)
2. **V2X Communication** (with latency/loss)
3. **Consensus Algorithm** (if multi-agent)
4. **Eco-TES Prediction** (battery envelope)
5. **Co-ESDRL Action** (brake blending)
6. **HOCBF Safety Filter** (QP solver)
7. **Vehicle Dynamics Update** (RK4 integration)
8. **Energy Accounting** (SOC update)
9. **Data Logging**

---

## 7. Euro NCAP Test Scenarios

### CCRs (Car-to-Car Rear Stationary)

**Initial Conditions**:
```
v_ego(0) = [10, 30, 50, 70] km/h
v_target(0) = 0 km/h
d_initial = d_brake + d_reaction
```

Where:
```
d_brake = v_ego² / (2 · a_brake)
d_reaction = v_ego · t_react
```

### CCRm (Car-to-Car Rear Moving)

**Initial Conditions**:
```
v_ego(0) = 50 km/h
v_target(0) = 20 km/h
v_rel(0) = 30 km/h
d_initial = calculated for TTC = 2.0 s
```

### CCRb (Car-to-Car Rear Braking)

**Initial Conditions**:
```
v_ego(0) = 50 km/h
v_target(0) = 50 km/h (then brakes at t=1s with -6 m/s²)
d_initial = 12 m (following distance)
```

### CPNCO-50 (Pedestrian Scenario)

**Initial Conditions**:
```
v_ego(0) = [20, 40, 60] km/h
pedestrian appears at t=0 from side obstruction
lateral offset = 1.0 m from vehicle path
pedestrian speed = 5 km/h (constant)
```

---

## 8. Consensus Algorithm (Multi-Agent)

### Trust Score Update

```
T_ij(t+1) = γ · T_ij(t) + (1-γ) · R_ij(t)
```

Where:
```
R_ij(t) = 1 / (1 + exp(β · (D_ij(t) - θ_rep)))
```

And:
```
D_ij(t) = ||x̂_i(t) - x̂_j(t)|| / max_k ||x̂_k(t) - μ_cluster(t)||
```

Parameters: γ = 0.85, β = 2.0, θ_rep = 0.5

### Weight Computation

```
τ_ij(t) = exp(-λ · ||x̂_i(t) - x̂_j(t)||²) · T_ij(t)
w_ij(t) = τ_ij(t) / (ε + Σ_k τ_ik(t))
```

Parameter: λ = 0.07

### Consensus State

```
x̄_i(t+1) = (1-μ) · Σ_{j∈N_i} w_ij(t) · x̂_j(t) + μ · x̄_i(t)
```

Parameter: μ = 0.15 (momentum)

### Byzantine Detection

```
D_M(i) = (x̂_i - μ_robust)ᵀ · Σ_robust^{-1} · (x̂_i - μ_robust)

Byzantine_flag = {
    1, if D_M(i) > χ²_{α,df}
    0, otherwise
}
```

Parameter: α = 0.05 (significance level), df = 8 (state dimension)

---

## 9. Statistical Analysis

### Performance Metrics

**Mean and Standard Deviation**:
```
μ = (1/N) · Σ[i=1→N] x_i
σ = sqrt((1/(N-1)) · Σ[i=1→N] (x_i - μ)²)
```

**Confidence Interval** (95%):
```
CI = μ ± t_{α/2, N-1} · (σ / sqrt(N))
```

Where t_{α/2, N-1} is the t-distribution critical value.

### Statistical Significance

**Two-Sample t-test**:
```
t = (μ_1 - μ_2) / sqrt(σ_1²/N_1 + σ_2²/N_2)
```

Null hypothesis H_0: μ_1 = μ_2 (no difference)

Reject H_0 if p < 0.01 (99% confidence).

---

## Implementation Notes

### Numerical Stability

1. **Avoid division by zero**: Add small epsilon (ε = 10^-6) to denominators
2. **Angle wrapping**: Keep θ ∈ [0, 2π) using modulo operation
3. **SOC bounds**: Clip SOC to [0, 1] after each update
4. **Temperature bounds**: Limit T_batt to [0°C, 60°C]

### Computational Efficiency

1. **Pre-compute constants**: Calculate fixed values once
2. **Vectorize operations**: Use NumPy/PyTorch batch operations
3. **Sparse updates**: Only update changed values in consensus
4. **Early termination**: Stop simulation when collision or stop detected

---

**Mathematical Framework**: DK  
**Platform Owner**: Gallop Holding Company  
**Institution**: Hong Kong Polytechnic University, EEE  
**Contact**: david.ko@connect.polyu.hk

**Note**: These mathematical models are complete and sufficient for independent implementation. Third-party developers can use these equations to build compatible simulation environments.

