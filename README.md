# Overview

This repository implements the solution of the VLDBJ paper **Table integration in data lakes unleashed: pairwise integrability judgment, integrable set discovery, and multi-tuple conflict resolution**

## Dataset
The dataset is available in the following link:
https://drive.google.com/drive/folders/1UUaP2z12o7Q-v535YyHoP_Xv6_EZhRc2?usp=sharing

## Models

### 1. SSACL (Self-Supervised Adversarial Contrastive Learning)
- **Purpose**: Pairwise integrability judgment
- **Location**: `src/pairwise_integrability/`
- **Description**: Determines if two tuples are integrable (refer to the same real-world entity)

### 2. Integrable Set Discovery
- **Purpose**: Find sets of integrable tuples
- **Location**: `src/integrable_set_discovery.py`
- **Methods**:
  - Bron-Kerbosch (Clique-based)
  - Louvain (Community detection)
  - Spectral Clustering
  - GNN (Graph Neural Network)

### 3. ICLCR (In-Context Learning for Conflict Resolution)
- **Purpose**: Resolve conflicts within integrable sets
- **Location**: `src/ICLCR/`
- **Description**: Uses LLM-based in-context learning to resolve attribute conflicts

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA (optional, for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Robin
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. (Optional) Install Infomap for additional community detection:
```bash
pip install infomap
```

## Usage

### Quick Start

Run the complete evaluation pipeline with default settings:

```bash
python main.py
```

### Advanced Options

```bash
# Specify output directories
python main.py --results-dir my_results --plots-dir my_plots

# Disable plot generation
python main.py --no-plots

# Set number of training epochs
python main.py --epochs 50

# Use custom configuration file
python main.py --config config.json
```

### Configuration

Create a custom `config.json` file:

```json
{
  "ssacl": {
    "pretrained_model": "microsoft/deberta-v3-base",
    "embedding_dim": 768,
    "epochs": 30,
    "learning_rate": 0.0001
  },
  "integrable_set": {
    "method": "bron_kerbosch",
    "n_clusters": null
  },
  "iclcr": {
    "k_demonstrations": 10,
    "mi_threshold": 0.1,
    "selection_strategy": "weighted_knn"
  },
  "evaluation": {
    "save_results": true,
    "generate_plots": true,
    "results_dir": "results",
    "plots_dir": "plots"
  }
}
```

## Output

The pipeline generates:

1. **Results Directory** (default: `results/`):
   - CSV files with evaluation metrics
   - LaTeX tables for publication
   - JSON files with detailed results
   - `summary_report.txt` - Comprehensive summary
   - `config.json` - Configuration used for the run

2. **Plots Directory** (default: `plots/`):
   - F1 score comparisons
   - Similarity score visualizations
   - Accuracy comparisons
   - Efficiency metrics
   - Hyperparameter heatmaps

## Evaluation Metrics

### Pairwise Integrability (SSACL)
- Precision, Recall, F1 Score
- Inference time
- Samples per second

### Integrable Set Discovery
- Similarity score (Jaccard-based)
- Number of sets found
- Execution time

### Conflict Resolution (ICLCR)
- Accuracy
- Average time per conflict
- Total conflicts resolved

## Project Structure

```
Robin/
├── main.py                 # Main pipeline script
├── evaluation.py           # Evaluation functions
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── src/                   # Source code
│   ├── __init__.py
│   ├── ICLCR/            # Conflict resolution model
│   │   ├── __init__.py
│   │   └── model.py
│   ├── pairwise_integrability/  # Pairwise judgment model
│   │   ├── __init__.py
│   │   ├── ssacl.py
│   │   ├── encoder.py
│   │   ├── matcher.py
│   │   ├── adversarial_trainer.py
│   │   └── data_generator.py
│   ├── integrable_set_discovery.py  # Set discovery methods
│   └── utils/            # Utility functions
│       ├── __init__.py
│       └── data_augmentation.py
├── results/              # Output results (generated)
└── plots/                # Visualization plots (generated)
```

## Contact
If you have any issue, please feel free to drop me at daomin.ji@student.rmit.edu.au

