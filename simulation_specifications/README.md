# CoPEM Framework - Simulation Platform Specifications

## Platform Ownership

The complete simulation platform used in this research is proprietary to **Gallop Holding Company's Advanced Mobility Research Laboratory**. This repository provides the specifications, parameters, and mathematical models necessary for third-party implementation, but does not include the full simulation source code.

## Purpose

These specifications enable:
- Independent reproduction of experimental results
- Third-party validation of the CoPEM framework
- Development of compatible simulation environments
- Integration with existing automotive simulation platforms

---

## Simulation Platform Overview

### Platform Architecture

The simulation environment consists of:

1. **Vehicle Dynamics Simulator** - 3-DOF kinematic model with Pacejka tire forces
2. **Battery Thermal Model** - Second-order Thévenin equivalent circuit
3. **V2X Network Simulator** - IEEE 802.11p DSRC protocol emulation
4. **Sensor Fusion Module** - GPS/IMU/RADAR data processing
5. **Traffic Scenario Generator** - Euro NCAP test protocol implementation

### Computing Infrastructure

- **Primary Server**: Intel Core i9-13900K, 2× NVIDIA RTX 4090
- **Simulation Framework**: Python 3.10 + PyTorch 2.0.1
- **Real-time Constraint**: 100 Hz control loop (10 ms timestep)
- **Parallel Processing**: Up to 16 scenarios simultaneously

---

## Contact

For licensing inquiries regarding the complete simulation platform:

**Platform Owner**: Gallop Holding Company  
**Research Contact**: DK (david.ko@connect.polyu.hk)  
**Institution**: Hong Kong Polytechnic University, EEE

---

**Note**: All mathematical models and parameters in this folder are sufficient for independent implementation. Third-party developers can build compatible simulation environments using these specifications.

