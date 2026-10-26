# 🧬 Drug Discovery Using LSTM-Based Molecular Generation and Multi-Agent Evaluation Framework

[![Conference](https://img.shields.io/badge/Conference-ICECAA%202026-blue.svg)](https://icecaa.com/2026/)
[![Publisher](https://img.shields.io/badge/Proceedings-Springer-orange.svg)](https://www.springer.com)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

An end-to-end de novo molecular generation and multi-agent evaluation framework combining a **2-Layer Stacked SELFIES-LSTM** generator with **SAFE+** (Sequential ADMET and Synthesizability Filtering Engine) to generate, filter, and prioritize novel, drug-like lead candidates with validated 3D conformational viability and biological target complementarity.

> **Accepted for Publication:**  
> **4th International Conference on Edge Computing and Applications (ICECAA 2026)**  
> Gnanamani College of Technology, Namakkal, Tamilnadu, India | December 4–5, 2026  
> *To be published in Springer Proceedings.*

---

## 👥 Authors & Affiliation

- **Harshitha M.\*** (`harshitham.btech23@rvu.edu.in`) — *Corresponding Author*
- **Kavya Garg** (`kavyag.btech23@rvu.edu.in`)
- **Thanyaa S.** (`thanyaas.btech23@rvu.edu.in`)
- **Madderla Chiranjeevi** (`madderlachiranjeevi@rvu.edu.in`)

*School of Computer Science and Engineering, RV University, Bengaluru, India*

---

## 📋 Table of Contents
- [Overview](#-overview)
- [SAFE+ Pipeline Architecture](#-safe-pipeline-architecture)
- [Dataset](#-dataset)
- [Five-Stage SAFE+ Multi-Agent Filtering](#-five-stage-safe-multi-agent-filtering)
- [Key Experimental Results](#-key-experimental-results)
- [The 2D Filter Blindspot](#-the-2d-filter-blindspot)
- [Benchmark Comparison with MOSES CharRNN](#-benchmark-comparison-with-moses-charrnn)
- [Target-Specific Molecular Docking (EGFR Kinase)](#-target-specific-molecular-docking-egfr-kinase)
- [Repository Structure](#-repository-structure)
- [Quickstart & Execution](#-quickstart--execution)
- [Citation](#-citation)

---

## 🔬 Overview

Most sequence-based molecular generative models report simple 2D topological validity and structural novelty, but rarely quantify downstream pharmacological attrition or 3D conformational strain liabilities. Furthermore, early SMILES-based generators suffer from syntax fragility (resulting in validity rates as low as 3.6% in unconstrained setups) and frequently rely on unvalidated heuristics (such as sequence length as a proxy for toxicity).

This project addresses these challenges by:
1. **Decoupling Generation from Syntax Fragility:** Utilizing **SELFIES (Self-Referencing Embedded Strings)** to mathematically guarantee **100.0% bond-topological validity** across all sampling temperatures ($T \in \{0.2, 0.5, 0.7, 1.0\}$).
2. **Instrumenting Sequential Attrition (SAFE+):** Tracking molecular retention through five non-negotiable hard-exclusion agents: 2D geometric bounds, PAINS screening, Lipinski/Veber pharmacokinetic rules, synthetic accessibility (SA Score $\le 4.5$) with SMARTS strain exclusion, and 3D MMFF94 force-field conformational strain relaxation.
3. **Uncovering the 2D Filter Blindspot:** Proving that standard 2D Lipinski/Veber rules permit up to **73.0% of candidates at $T{=}1.0$ that harbor severe synthetic complexity or unrelaxed 3D geometric ring strain**.
4. **Target-Specific Validation:** Confirming biological target complementarity via in silico docking against the human EGFR kinase domain (PDB: 1M17), achieving binding affinities competitive with clinical inhibitors.

---

## 🏗️ SAFE+ Pipeline Architecture

```
ZINC250K Database (249,455 SMILES)
         │
         ▼
┌─────────────────────────────────┐
│  Data Preprocessing             │  → Canonicalization via RDKit
│  & Vocabulary Construction      │  → Conversion to 34-token SELFIES (L_max = 80)
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  2-Layer Stacked LSTM           │  → 2× 256 hidden units, 20% Dropout, Adam (lr=1e-3)
│  Training (30 epochs)           │  → Best validation loss: 1.156, Token acc: ~64%
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Temperature-Scaled Sampling    │  → T ∈ {0.2, 0.5, 0.7, 1.0} across seeds 42, 123, 456
│  SELFIES → Graph Decoding       │  → 100.0% bond-topological validity guaranteed
└─────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────────────┐
│  SAFE+ Five-Stage Sequential Evaluation Engine                         │
│                                                                        │
│  Stage 1: Geometric Hard Filter  → MW ∈ [120, 600] Da, Heavy ≥ 10      │
│  Stage 2: PAINS Screen           → Baell & Holloway Catalogue (RDKit)  │
│  Stage 3: Lipinski / Veber 2D    → logP ≤ 5, HBD ≤ 5, HBA ≤ 10, TPSA   │
│  Stage 4: Synthesizability/Strain→ Ertl SA Score ≤ 4.5, SMARTS strain  │
│  Stage 5: 3D MMFF94 Force Field  → ΔE_strain ≤ 35.0 kcal/mol           │
└────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Prioritized Lead Selection     │  → QED - normalized ADMET penalty vector
│  & In Silico Docking (EGFR)     │  → AutoDock Vina / PyRx (PDB: 1M17)
└─────────────────────────────────┘
```

---

## 📊 Dataset

- **Name:** ZINC250K Clean Commercial Subset
- **Source:** [ZINC Database / Kaggle ZINC250K](https://www.kaggle.com/datasets/basu369victor/zinc250k)
- **Size:** 249,455 valid drug-like small molecules
- **Split:** Deterministic 90% training (224,510 compounds) / 10% validation (24,945 compounds)
- **Token Vocabulary:** 34 unique SELFIES tokens + special tokens (`<SOS>`, `<EOS>`, `<PAD>`)
- **Seeds:** 42, 123, and 456 for training and sampling verification

---

## 🧪 Five-Stage SAFE+ Multi-Agent Filtering

| Stage | Filtering Agent | Hard Rule / Criterion | Objective |
|:-----:|:----------------|:----------------------|:----------|
| **1** | Geometric Bounds | $\text{MW} \in [120, 600]$ Da, $N_{\text{heavy}} \ge 10$, $N_{\text{rot}} \le 10$ | Purge low-mass fragments and excessively flexible chains |
| **2** | PAINS Screen | Baell & Holloway substructure catalogue | Eliminate false-positive promiscuous assay disruptors |
| **3** | Lipinski \& Veber | $\log P \le 5.0$, $\text{HBD} \le 5$, $\text{HBA} \le 10$, $\text{TPSA} \le 140$ \AA$^2$ | Ensure drug-likeness and oral bioavailability windows |
| **4** | Synthesizability \& Strain | Ertl SA Score $\le 4.5$; exclude cumulenes ($C{=}C{=}C$), allenes, small-ring alkynes ($<8$ atoms) | Purge synthetically intractable structures and explicit 2D ring strain |
| **5** | 3D MMFF94 Relaxation | Distance geometry embed $\rightarrow$ MMFF94 minimize $\rightarrow \Delta E_{\text{strain}} \le 35.0$ kcal/mol | Purge 3D steric clashes, non-bonded overlaps, and unrelaxed conformations |

### Attrition Funnel at $T{=}0.5$ ($n{=}999$ Raw Sequences)
- **Stage 0 (Raw Generated):** 999
- **Stage 1 (Syntactic Validity):** 999 (**100.0%** validity)
- **Stage 2 (Geometric Bounds):** 417 (**41.7%** retention)
- **Stage 3 (PAINS Screen):** 402 (**96.4%** pass rate)
- **Stage 4 (Lipinski/Veber 2D):** 369 (**91.8%** pass rate)
- **Stage 5 (SA Score \& Strain Filter):** 205 (**55.6%** pass rate)
- **Final Unique Retained Leads:** **199 unique candidates** (**19.9%** net retention rate)

---

## 📈 Key Experimental Results

### Core Benchmark Metrics at $T{=}0.5$ (Retained Candidate Set)
- **Unique Candidates Retained:** 199 molecules
- **Topological Validity:** 100.0%
- **Chemical Novelty (Tanimoto $< 0.40$ vs. 249,455 ZINC250K compounds):** 100.0% (mean nearest-neighbor: 0.216)
- **PAINS-Free Rate:** 96.3%
- **Synthetic Accessibility (Mean SA Score):** 3.38 (highly accessible)
- **Mean QED:** 0.559
- **Internal Diversity:** 0.896
- **Mean MMFF94 Strain Energy:** 50.43 kcal/mol
- **5D Joint Property Wasserstein Distance ($WD_{\text{5D}}$):** 0.1770
- **Bemis-Murcko Scaffold Shannon Entropy ($H_{\text{scaffold}}$):** 6.31 bits

---

## ⚠️ The 2D Filter Blindspot

A primary empirical contribution of this benchmark is discovering the **2D filter blindspot**:
- At high sampling temperature ($T = 1.0$), **93.9%** of generated molecules pass standard 2D Lipinski Rule of 5 and Veber rules.
- However, when evaluated under SAFE+ Stage 4 (SA Score \& SMARTS strain) and Stage 5 (MMFF94 3D force field), **73.0% of these 2D-compliant candidates fail**.
- The failures stem from cumulated double bonds ($C{=}C{=}C$), strained small-ring alkynes, and severe 3D steric clashes.
- Furthermore, the mean MMFF94 strain energy among surviving candidates escalates **2.14-fold** (from 50.43 kcal/mol at $T=0.5$ to 108.15 kcal/mol at $T=1.0$).
- **Conclusion:** Standard 2D rules create a false sense of chemical viability at high generation entropy; 3D force-field and synthesizability screening must accompany 2D filters.

---

## 📊 Benchmark Comparison with MOSES CharRNN

| Generative Configuration | Validity | Uniqueness | Novelty | Drug-like | PAINS-free | Avg QED | Int. Div. | Scaf. Div. |
|:-------------------------|:--------:|:----------:|:-------:|:---------:|:----------:|:-------:|:---------:|:----------:|
| **$T = 0.2$ (SELFIES)** | 100.0% | 87.1% | 99.2% | 99.2% | 93.6% | 0.547 | 0.867 | 0.707 |
| **$T = 0.5$ (SELFIES, Pareto)** | **100.0%** | **98.4%** | **98.7%** | **99.7%** | **96.3%** | **0.559** | **0.896** | **0.820** |
| **$T = 0.7$ (SELFIES)** | 100.0% | 100.0% | 97.9% | 99.5% | 96.3% | 0.551 | 0.909 | 0.872 |
| **$T = 1.0$ (SELFIES)** | 100.0% | 100.0% | 99.8% | 98.7% | 93.4% | 0.504 | 0.930 | 0.854 |
| Random ZINC250K Baseline | 99.1% | 89.6% | 0.0% | 100.0% | 97.5% | 0.738 | 0.879 | 0.806 |
| **MOSES CharRNN Baseline** | 97.3% | 99.9% | 84.2% | N/A | N/A | 0.600 | 0.856 | 0.887 |

*Our SELFIES-LSTM model achieves **100% validity** (vs. 97.3%), outperforms CharRNN on **novelty by +14.5%** (98.7% vs. 84.2%), and exhibits superior **Internal Diversity** (0.896 vs. 0.856).*

---

## 🎯 Target-Specific Molecular Docking (EGFR Kinase)

To evaluate biological activity beyond 2D empirical heuristics, the top unstrained candidate was docked into the ATP-binding pocket of human **EGFR kinase** (PDB ID: [1M17](https://www.rcsb.org/structure/1M17)) using AutoDock Vina / PyRx:

- **Lead Candidate:** $N$-(2-(2-(2-fluorophenyl)cyclopropyl)ethyl)acetamide (Lead-3)
- **SMILES:** `CC(=O)NCCC1CC1c1ccccc1F`
- **Physicochemical Profile:** MW: 221.28 Da | $\log P$: 2.46 | TPSA: 29.10 \AA$^2$ | QED: 0.831 | SA Score: 2.87 | Tox Penalty: 0.00 | $S_{\text{final}}$: 0.916
- **Predicted Binding Affinity:** **$-7.6$ kcal/mol** (competitive with clinical reference Erlotinib at $-8.1$ kcal/mol)
- **Residue-Level SAR:**
  - Acetamide carbonyl oxygen forms a direct **hydrogen bond with the Met793 hinge residue** (distance: 2.1 \AA).
  - 2-fluorophenyl group occupies the deep **hydrophobic subpocket lined by Leu718, Val726, and Ala743**.
  - Cyclopropyl linker restricts rotatable bond entropy ($N_{\text{rot}} = 4$), stabilizing the bioactive conformation without 3D geometric strain.

---

## 📁 Repository Structure

```
├── data/                                   # Dataset storage
│   └── 250k_rndm_zinc_drugs_clean_3.csv   # Curated ZINC250K SMILES library
├── figures/                                # High-resolution publication plots
│   ├── 01_training_loss_curve.png         # Loss and token accuracy over 30 epochs
│   ├── 02_metrics_comparison.png          # Baseline metric comparisons
│   ├── 03_temperature_sweep.png           # 6-metric temperature ablation
│   ├── 04_property_distributions.png      # Physicochemical histograms (MW, logP, TPSA, QED)
│   ├── 05_toxicity_breakdown.png          # ADMET penalty component decomposition
│   ├── 06_top12_molecules.png             # 2D structural panels of top-12 leads
│   ├── 07_radar_top5.png                  # Multi-axis pharmacological radar chart
│   ├── 08_correlation_heatmap.png         # Pearson correlation matrix (9 properties)
│   ├── 09_novelty_tanimoto_hist.png       # Tanimoto similarity histogram vs. ZINC250K
│   ├── 10_docking_pose.png                # PyRx docking pose in EGFR ATP-binding site
│   └── 11_safe_attrition.png              # SAFE+ per-stage attrition funnel
├── models/                                 # Trained model weights & checkpoints (.pt)
├── results/                                # Filtered lead candidates (molecules_T*_SAFEplus.csv)
├── src/                                    # Source code pipeline
│   ├── config.py                          # Hyperparameters, paths, and thresholds
│   ├── data_loader.py                     # SMILES canonicalization and SELFIES tokenization
│   ├── evaluate.py                        # SAFE+ five-stage multi-agent evaluation engine
│   ├── generate.py                        # Temperature-controlled autoregressive sampling
│   ├── main.py                            # Execution entry point
│   ├── model.py                           # 2-layer stacked LSTM architecture
│   ├── train.py                           # Model training and validation loops
│   └── visualize.py                       # Plotting and figure generation scripts
├── requirements.txt                        # Python environment dependencies
├── run.py                                  # Pipeline runner script
└── README.md                               # Project documentation
```

---

## 🚀 Quickstart & Execution

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Kavya-Garg003/SAFE-An-Empirical-Benchmark-of-Multi-Stage-ADMET-Synthetizability-and-Structural-Strain-Attritio.git
cd SAFE-An-Empirical-Benchmark-of-Multi-Stage-ADMET-Synthetizability-and-Structural-Strain-Attritio

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Full Pipeline
Executes data preprocessing, model training (30 epochs), temperature sampling ($T \in \{0.2, 0.5, 0.7, 1.0\}$), SAFE+ 5-stage filtering, metric evaluation, and figure generation:
```bash
python run.py
```

### 3. Quick Verification Run
Runs a fast smoke-test with 5,000 molecules and 5 epochs to verify environment functionality:
```bash
python run.py --quick
```

---

## 📜 Citation

If you find this codebase or benchmark useful in your research, please cite our paper:

```bibtex
@inproceedings{harshitha2026drug,
  title={Drug Discovery Using LSTM-Based Molecular Generation and Multi-Agent Evaluation Framework},
  author={Harshitha, M. and Garg, Kavya and Thanyaa, S. and Chiranjeevi, Madderla},
  booktitle={Proceedings of the 4th International Conference on Edge Computing and Applications (ICECAA 2026)},
  year={2026},
  publisher={Springer}
}
```
