# Vehicle Dynamics Parameters

## Ego Vehicle (Electric Sedan)

### Physical Dimensions

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| Mass (m) | 1850 | kg | Including battery pack |
| Wheelbase (l) | 2.9 | m | Front to rear axle |
| Front axle distance (l_f) | 1.3 | m | CoG to front axle |
| Rear axle distance (l_r) | 1.6 | m | CoG to rear axle |
| Yaw inertia (I_z) | 3200 | kg·m² | About vertical axis |
| CoG height | 0.55 | m | Center of gravity |
| Track width | 1.58 | m | Lateral wheel separation |

### Aerodynamics

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| Frontal area (A_front) | 2.3 | m² | Projected area |
| Drag coefficient (C_d) | 0.28 | - | Aerodynamic drag |
| Air density (ρ_air) | 1.225 | kg/m³ | Standard atmosphere |
| Lift coefficient (C_L) | 0.15 | - | Downforce generation |

### Tire Parameters (Pacejka Model)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Wheel radius (r_wheel) | 0.33 | m |
| B_x (stiffness factor) | 10.0 | Longitudinal slip |
| C_x (shape factor) | 1.9 | Slip curve shape |
| D_x (peak factor) | F_z × μ_peak | Maximum force |
| E_x (curvature factor) | 0.97 | Curve curvature |
| μ_peak (dry asphalt) | 0.9 | Peak friction coefficient |
| μ_peak (wet asphalt) | 0.7 | Wet condition |
| Rolling resistance (C_r) | 0.01 | - |

### Brake System

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| Max friction brake force | 16000 | N | Mechanical limit |
| Max friction deceleration | 9.0 | m/s² | At full application |
| Brake response time | 50 | ms | Hydraulic actuation |
| ABS activation threshold | 0.15 | - | Slip ratio limit |
| Friction brake efficiency | 0.92 | - | Mechanical efficiency |

### Regenerative Braking

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| Max regen power | 60 | kW | Motor limit |
| Max regen torque | 250 | N·m | At motor shaft |
| Regen efficiency | 0.85 | - | Motor + inverter |
| Min speed for regen | 5 | km/h | Below this, friction only |
| Motor gear ratio | 9.73 | - | Single-speed transmission |

---

## Battery Pack

### Electrical Specifications

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| Capacity | 75 | kWh | Total energy |
| Nominal voltage | 350 | V | Pack voltage |
| Cell configuration | 96s74p | - | Series × Parallel |
| Max charge current | 400 | A | Peak charging |
| Max discharge current | 500 | A | Peak power output |
| Internal resistance (R_0) | 0.05 | Ω | At 25°C, 50% SOC |

### Thermal Parameters

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| Battery mass (m_batt) | 500 | kg | Pack weight |
| Specific heat (c_p) | 1000 | J/(kg·K) | Thermal capacity |
| Heat transfer coeff (h) | 10 | W/(m²·K) | Convection |
| Surface area (A_surf) | 4.0 | m² | Heat exchange |
| Thermal resistance (R_th) | 0.5 | K/W | Pack to ambient |
| Operating temp range | 15-45 | °C | Safe operation |
| Optimal temp range | 20-35 | °C | Best performance |

### SOC-Dependent Characteristics

| SOC (%) | Voltage (V) | R_0 (Ω) | Max Regen (kW) |
|---------|-------------|---------|----------------|
| 10 | 320 | 0.08 | 30 |
| 20 | 335 | 0.06 | 45 |
| 50 | 350 | 0.05 | 60 |
| 80 | 365 | 0.06 | 50 |
| 90 | 370 | 0.08 | 35 |
| 95 | 372 | 0.12 | 20 |

---

## Sensor Specifications

### GPS/IMU (Inertial Navigation)

| Sensor | Noise (σ, 1σ) | Update Rate | Description |
|--------|---------------|-------------|-------------|
| GPS position | 0.5 m | 10 Hz | Horizontal accuracy |
| GPS velocity | 0.1 m/s | 10 Hz | Speed measurement |
| IMU acceleration | 0.05 m/s² | 100 Hz | Linear acceleration |
| IMU heading | 0.5° | 100 Hz | Yaw angle |
| IMU yaw rate | 0.01 rad/s | 100 Hz | Angular velocity |

### RADAR (Front-facing)

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| Range noise (σ_r) | 0.3 | m | Distance measurement |
| Range rate noise (σ_v) | 0.2 | m/s | Relative velocity |
| Azimuth noise (σ_az) | 1.0 | ° | Horizontal angle |
| Max range | 200 | m | Detection limit |
| Field of view | 80 | ° | Horizontal coverage |
| Detection probability | 0.98 | - | At SNR > 15 dB |
| False alarm rate | 10^-6 | - | Per scan |
| Update rate | 20 | Hz | Measurement frequency |

---

## V2X Communication (IEEE 802.11p)

### Radio Parameters

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| Frequency | 5.9 | GHz | DSRC Channel 172 |
| Bandwidth | 10 | MHz | Channel bandwidth |
| Transmit power | 20 | dBm | EIRP |
| Antenna gain | 3 | dBi | Omnidirectional |
| Receiver sensitivity | -85 | dBm | Detection threshold |

### Channel Model

**Path Loss** (log-distance):
```
RSSI(d) = P_tx - 10·n·log₁₀(d) - X_σ - L_vehicle
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| Path loss exponent (n) | 2.7 | Suburban environment |
| Shadowing std (σ) | 4 | dB |
| Vehicle body loss (L_vehicle) | 10 | dB |

**Packet Error Rate**:
- RSSI > -70 dBm: PER = 2%
- -80 to -70 dBm: PER = 10%
- -90 to -80 dBm: PER = 30%
- RSSI < -90 dBm: PER = 60%

### Communication Protocol

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| Message type | CAM | - | Cooperative Awareness |
| Message rate | 10 | Hz | Broadcast frequency |
| Message size | 300-500 | bytes | Typical payload |
| Max latency | 100 | ms | Protocol timeout |
| Communication range | 50-300 | m | Environment dependent |

---

## Road and Environment

### Road Surface

| Condition | μ_peak | Description |
|-----------|--------|-------------|
| Dry asphalt | 0.9 | Standard test |
| Wet asphalt | 0.7 | Rain condition |
| Icy road | 0.3 | Winter scenario |

### Grade

| Parameter | Value | Unit |
|-----------|-------|------|
| Test grade | 0 | % |
| Max simulated grade | ±10 | % |

---

## Implementation Notes

These parameters were calibrated using:
- Tesla Model 3 telemetry data (100 driving cycles)
- Laboratory battery cell testing (1000+ charge/discharge cycles)
- RADAR manufacturer specifications (Continental ARS540)
- IEEE 802.11p field measurements (Urban/suburban Hong Kong)

All parameters are provided in SI units unless otherwise specified.

---

**Proprietary Platform**: Gallop Holding Company  
**Research Institution**: Hong Kong Polytechnic University, EEE  
**Contact**: DK (david.ko@connect.polyu.hk)

