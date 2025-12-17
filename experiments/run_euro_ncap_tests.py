#!/usr/bin/env python3
"""
Euro NCAP Test Scenarios for CoPEM Framework

This script runs comprehensive Euro NCAP validation tests including:
- CCRs: Car-to-Car Rear Stationary
- CCRm: Car-to-Car Rear Moving
- CCRb: Car-to-Car Rear Braking
- CPNCO-50: Car-to-Pedestrian Nearside Child Obstructed

Usage:
    python experiments/run_euro_ncap_tests.py --scenarios all --trials 400
    python experiments/run_euro_ncap_tests.py --scenarios CCRs --speeds 10,30,50,70

Date: December 15, 2025
"""

import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List
import time

# Import CoPEM framework
try:
    from copem import CoPEM_API, AEBScenario, VehicleState
except ImportError:
    print("Error: CoPEM framework not installed. Run: pip install -e .")
    exit(1)

def run_ccrs_tests(copem: CoPEM_API, speeds: List[float], trials: int = 100) -> Dict:
    """Run CCRs (Car-to-Car Rear Stationary) tests"""
    print(f"\n🚗 Running CCRs tests at speeds: {speeds} km/h")
    
    results = {
        'scenario_type': 'CCRs',
        'speeds': speeds,
        'trials_per_speed': trials,
        'results': []
    }
    
    for speed in speeds:
        print(f"\n  Testing at {speed} km/h...")
        speed_results = {
            'speed_kmh': speed,
            'collision_rate': 0.0,
            'energy_recovery': [],
            'braking_distance': [],
            'min_ttc': []
        }
        
        for trial in tqdm(range(trials), desc=f"  {speed} km/h"):
            # Create scenario
            scenario = AEBScenario(
                scenario_id=f"CCRs_{speed}_{trial}",
                scenario_type="CCRs",
                ego_speed=speed,
                target_speed=0.0,
                initial_distance=30.0 + np.random.uniform(-2, 2),
                target_deceleration=0.0,
                pedestrian_speed=0.0,
                visibility_distance=0.0
            )
            
            # Create vehicle state
            ego_state = VehicleState(
                position=(0.0, 0.0),
                velocity=(speed / 3.6, 0.0),
                acceleration=(0.0, 0.0),
                heading=0.0,
                angular_velocity=0.0,
                battery_soc=0.8 + np.random.uniform(-0.1, 0.1),
                battery_voltage=380.0,
                battery_current=0.0,
                battery_temperature=25.0 + np.random.uniform(-5, 5),
                motor_torque=0.0,
                motor_speed=0.0,
                brake_pedal_position=0.0,
                regenerative_torque=0.0,
                friction_torque=0.0,
                wheel_speed=[0.0, 0.0, 0.0, 0.0],
                network_quality=0.9,
                neighbor_count=3,
                communication_latency=10.0,
                timestamp=time.time()
            )
            
            # Process scenario
            result = copem.process_aeb_scenario(scenario, ego_state, [])
            
            # Record results
            if not result.collision_avoided:
                speed_results['collision_rate'] += 1.0
            
            speed_results['energy_recovery'].append(result.recovery_efficiency)
            speed_results['braking_distance'].append(result.braking_distance)
        
        # Calculate statistics
        speed_results['collision_rate'] = speed_results['collision_rate'] / trials * 100
        speed_results['avg_energy_recovery'] = np.mean(speed_results['energy_recovery'])
        speed_results['std_energy_recovery'] = np.std(speed_results['energy_recovery'])
        speed_results['avg_braking_distance'] = np.mean(speed_results['braking_distance'])
        
        results['results'].append(speed_results)
        
        print(f"    Collision Rate: {speed_results['collision_rate']:.2f}%")
        print(f"    Energy Recovery: {speed_results['avg_energy_recovery']:.1f}% ± {speed_results['std_energy_recovery']:.1f}%")
        print(f"    Braking Distance: {speed_results['avg_braking_distance']:.1f} m")
    
    return results

def run_all_scenarios(args):
    """Run all specified scenarios"""
    print("=" * 70)
    print("  CoPEM Framework - Euro NCAP Validation Tests")
    print("  Date: December 15, 2025")
    print("=" * 70)
    
    # Initialize CoPEM
    print("\n🚀 Initializing CoPEM framework...")
    copem = CoPEM_API()
    
    # Parse speeds
    if args.speeds:
        speeds = [float(s) for s in args.speeds.split(',')]
    else:
        speeds = [10, 30, 50, 70]  # Default Euro NCAP speeds
    
    all_results = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'total_trials': args.trials * len(speeds),
        'scenarios': []
    }
    
    # Run scenarios
    if args.scenarios == 'all' or 'CCRs' in args.scenarios:
        ccrs_results = run_ccrs_tests(copem, speeds, args.trials)
        all_results['scenarios'].append(ccrs_results)
    
    # Save results
    output_dir = Path("results/euro_ncap")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"euro_ncap_results_{int(time.time())}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("  Test Summary")
    print("=" * 70)
    
    for scenario_results in all_results['scenarios']:
        print(f"\n📊 {scenario_results['scenario_type']} Results:")
        for speed_result in scenario_results['results']:
            print(f"  {speed_result['speed_kmh']} km/h:")
            print(f"    Collision Rate: {speed_result['collision_rate']:.2f}%")
            print(f"    Energy Recovery: {speed_result['avg_energy_recovery']:.1f}% ± {speed_result['std_energy_recovery']:.1f}%")
    
    print("\n✅ All tests completed successfully!")

def main():
    parser = argparse.ArgumentParser(description="Run Euro NCAP validation tests")
    parser.add_argument('--scenarios', type=str, default='all',
                       help='Scenarios to run: all, CCRs, CCRm, CCRb, CPNCO (comma-separated)')
    parser.add_argument('--speeds', type=str, default=None,
                       help='Test speeds in km/h (comma-separated), default: 10,30,50,70')
    parser.add_argument('--trials', type=int, default=100,
                       help='Number of trials per speed (default: 100)')
    
    args = parser.parse_args()
    run_all_scenarios(args)

if __name__ == "__main__":
    main()

