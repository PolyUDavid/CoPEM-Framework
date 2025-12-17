# JSON to Paper: Data Traceability Documentation

## Purpose

This folder provides complete traceability between the experimental data files and the results reported in our paper. I created this documentation to ensure full transparency and enable independent verification of all claims.

## Contents

### DATA_MAPPING.md

**Complete mapping documentation** showing exactly how each number in the paper was derived from the JSON data files.

Includes:
- Table-by-table mapping of paper results to JSON paths
- Calculation methods for all metrics
- Statistical significance testing procedures
- Data integrity verification
- Sample size justification

## Why This Matters

### For Reviewers

- Verify every claim in the paper by checking the JSON source
- Understand the statistical methodology
- Confirm sample sizes are adequate
- Check data integrity via checksums

### For Researchers

- Reproduce our exact results using the same data
- Understand how we computed each metric
- Apply similar methodology to your own work
- Build upon our research with confidence

### For Third Parties

- Independent validation of the CoPEM framework
- Transparency in experimental methodology
- Clear audit trail from raw data to published results

## Data Files Referenced

All mappings reference files in `data/paper_data/`:

1. **copem_paper_results.json** - Primary source for paper metrics
2. **copem_complete_experiment_results_20250714_151845.json** - Training data
3. **copem_case3_fleet_cooperative_results_20250714_172108.json** - Fleet scenarios
4. **copem_integrated_experiment_results_20250714_172137.json** - Integration tests

## Verification Process

To verify any claim in the paper:

1. Find the claim in `DATA_MAPPING.md`
2. Note the JSON path provided
3. Open the corresponding JSON file
4. Navigate to the specified path
5. Verify the value matches

### Example

**Paper Claim**: "36.5% energy recovery"

**Verification**:
```bash
# Open the primary data file
cat data/paper_data/copem_paper_results.json | jq '.core_achievements.single_vehicle_energy_recovery_percent'

# Output: 36.5
```

## Experimental Rigor

All experiments followed strict protocols:

- **Sample Sizes**: 1050+ trials for single-vehicle, 100 trials for fleet
- **Statistical Significance**: p < 0.01 for all comparisons
- **Reproducibility**: Fixed random seed (42), documented parameters
- **Standards Compliance**: Euro NCAP test protocols
- **Data Integrity**: SHA-256 checksums, version control

Full details in `DATA_MAPPING.md`.

## Contact

For questions about data mapping or verification:

**Author**: DK  
**Email**: david.ko@connect.polyu.hk  
**Institution**: Hong Kong Polytechnic University, EEE

---

**Note**: This documentation was maintained throughout the research process, not created retroactively. All data generation and analysis followed the methodology described in the paper.

