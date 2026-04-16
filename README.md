# NaviDiv: A Comprehensive Framework for Monitoring Chemical Diversity in Generative Molecular Design

**NaviDiv** is a comprehensive framework for analyzing chemical diversity in generative molecular design, with a focus on understanding how different diversity metrics evolve during reinforcement learning optimization. The framework introduces multiple complementary metrics that capture different aspects of molecular variation: representation distance-based, string-based, fragment-based, and scaffold-based approaches.

## Features

### Multiple Diversity Metrics

- **Representation Distance-Based**: Using molecular fingerprints (Morgan, RDKit) and similarity metrics (Tanimoto coefficient)
- **String-Based Analysis**: N-gram analysis of SMILES representations for sequence-level diversity assessment
- **Fragment-Based Metrics**: Systematic molecular decomposition using BRICS fragmentation and frequency analysis
- **Scaffold-Based Methods**: Bemis-Murcko scaffold analysis for core molecular framework comparison
- **Ring System Analysis**: Identification and analysis of ring systems and their sizes
- **Functional Group Analysis**: Detection and diversity assessment of functional groups

### Real-Time Monitoring & Visualization

- **Interactive Molecular Visualization**: 2D structural representations with sorting and filtering options
- **Temporal Analysis**: Monitor evolution of specific molecular fragments and cluster formation patterns
- **Chemical Space Projection**: t-SNE and PCA visualization of molecular diversity evolution
- **Comparative Analysis**: Similarity assessment against user-defined reference sets

### Integration Capabilities

- **REINVENT4 Compatible**: Seamless integration with reinforcement learning workflows
- **Real-Time Penalty Functions**: Adaptive diversity constraints during generation
- **Computational Efficiency**: Minimal overhead (~3 seconds per 100 molecules)
- **Statistical Analysis**: Comprehensive diversity trend reports with significance testing

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/mohammedazzouzi15/NaviDiv.git
cd NaviDiv
```

### 2. Create and Activate Conda Environment

```bash
conda create -n NaviDiv python==3.12
conda activate NaviDiv
```

### 3. Choose Installation Type

**Standard Installation (Core Framework)**

Install the core NaviDiv package with essential dependencies for diversity analysis:

```bash
pip install -e .
```

**Full Installation (with REINVENT4 Integration)**

For complete generative molecular design workflows with REINVENT4:

First, install PyTorch following the [official documentation](https://pytorch.org/get-started/locally/):

```bash
conda install pytorch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 pytorch-cuda=12.4 -c pytorch -c nvidia
```

Then install REINVENT4 and NaviDiv with full dependencies:

```bash
git clone https://github.com/mohammedazzouzi15/REINVENT4_div.git
cd REINVENT4_div
pip install --no-deps -e .
cd ../
pip install -e .[reinvent]
```

### 4. Optional Dependencies

For enhanced molecular manipulation capabilities:

```bash
conda install openeye::openeye-toolkits
```

## Quick Start

### Interactive Dashboard

Launch the Streamlit dashboard for comprehensive diversity analysis:

```bash
streamlit run app.py
```

### Programmatic Usage

```python
from navidiv.diversity.diversity import diversity_all
from rdkit import Chem

# Load SMILES strings
smiles_list = ["CCO", "CCN", "CCC"]  # Your SMILES data

# Calculate various diversity metrics
richness = diversity_all(smiles=smiles_list, mode="Richness")
internal_diversity = diversity_all(smiles=smiles_list, mode="IntDiv")
scaffold_diversity = diversity_all(smiles=smiles_list, mode="BM")

# Analyze functional groups and ring systems
functional_groups = diversity_all(smiles=smiles_list, mode="FG")
ring_systems = diversity_all(smiles=smiles_list, mode="RS")
```

### Integration with REINVENT4

```python
from navidiv.reinvent.run_staged_learning_2 import run_staged_learning
from navidiv.reinvent.InputGenerator import InputGenerator
from omegaconf import DictConfig

# Create configuration
cfg = DictConfig({...})  # Your REINVENT config

# Generate input files with diversity filters
input_generator = InputGenerator(cfg)
input_generator.generate_input()

# Run staged learning with diversity constraints
run_staged_learning(cfg)
```

---

## Using the NaviDiv App

> Full tutorial: [Tutorials/Using_The_app/app_tutorial.md](Tutorials/Using_The_app/app_tutorial.md)

The interactive Streamlit dashboard provides a no-code interface for exploring molecular diversity.

### Launching

```bash
conda activate NaviDiv
streamlit run app.py
# Opens at http://localhost:8501
```

### Input Data Format

Prepare a CSV file with:
- **`smiles`** column (required): SMILES strings of molecules
- **`step`** column (optional): generation step index, for temporal analysis
- **`Score`** column (optional): optimization score, for tracking RL progress

### Recommended Workflow

1. **Load dataset** — enter the full path to your CSV and click *Load File*
2. **Run t-SNE** — projects molecular fingerprints to 2D for chemical space visualization
3. **Run individual scorers** — choose from the sidebar:
   | Scorer | What it measures |
   |---|---|
   | Ngram | SMILES sequence pattern diversity |
   | Scaffold | Bemis-Murcko core framework diversity |
   | Cluster | Similarity-based molecular grouping |
   | RingScorer | Ring system diversity |
   | FGscorer | Functional group diversity |
   | Fragments_default | BRICS fragment decomposition |
4. **Run All Scorers** — comprehensive analysis across all metrics; outputs files to `scorer_output/`
5. **Interpret results** — two tabs:
   - **Per Fragment**: frequency bar plots of structural motifs
   - **Per Step**: diversity trends over optimization steps (requires `step` column)

### Temporal Evolution Analysis

If your CSV contains a `step` column (e.g., from a REINVENT4 run), the *Per Step* tab tracks how diversity evolves across the optimization. Look for:
- **Exploration phase**: high diversity early in RL
- **Exploitation phase**: diversity narrows as the model focuses
- **Convergence phase**: plateau around the optimum

---

## Running REINVENT4 with Diversity Constraints

> Full tutorial: [Tutorials/Running_Reinvent/reinvent_implementation_tutorial.md](Tutorials/Running_Reinvent/reinvent_implementation_tutorial.md)

NaviDiv adds diversity constraint scoring to REINVENT4 reinforcement learning. Instead of optimizing only for a target property, you balance property optimization with structural diversity.

### Environment Setup

```bash
conda activate reinvent4

# Verify installation
python -c "import navidiv; print('NaviDiv OK')"
python -c "import reinvent; print('REINVENT4 OK')"

# Set repo root (required for all commands below)
export NAVIDIV_ROOT="$(cd "$(git rev-parse --show-toplevel)" && pwd)"
```

### Pre-configured Diversity Strategies

Working configurations are in `Tutorials/Running_Reinvent/example/conf_folder/diversity_scorer/`:

| Configuration | Description | Best for |
|---|---|---|
| `All_constraints.yaml` | All diversity metrics, moderate constraints | Balanced exploration + optimization |
| `All_weak_constraints.yaml` | Light constraints, property-first | When diversity is secondary |
| `scaffold_only.yaml` | Scaffold diversity focus | Exploring different core frameworks |
| `fragement_only.yaml` | Fragment diversity focus | Exploring molecular building blocks |
| `ngram_only.yaml` | SMILES string pattern diversity | Varying SMILES sequence patterns |
| `similarity_only.yaml` | Cluster-based diversity | Preventing generation of near-duplicates |

### Running a Single Diversity Scorer

```bash
cd $NAVIDIV_ROOT/Tutorials/Running_Reinvent/example/
conda activate reinvent4
export PYTHONPATH="${PYTHONPATH}:$NAVIDIV_ROOT/src/navidiv/reinvent"

python3 $NAVIDIV_ROOT/src/navidiv/reinvent/run_reinvent_2.py \
    --config-name test \
    --config-path $NAVIDIV_ROOT/Tutorials/Running_Reinvent/example/conf_folder \
    name=scaffold_experiment \
    wd=./runs/scaffold_test \
    input_generator.file_path=./InputGenerator_custom.py \
    reinvent_common.prior_filename=./priors/formed.prior \
    reinvent_common.agent_filename=./priors/formed.prior \
    reinvent_common.max_steps=1000 \
    diversity_scorer=scaffold_only
```

For a quick test run (10 steps):

```bash
python3 $NAVIDIV_ROOT/src/navidiv/reinvent/run_reinvent_2.py \
    --config-name test \
    --config-path $NAVIDIV_ROOT/Tutorials/Running_Reinvent/example/conf_folder \
    name=test_run \
    wd=./runs/test \
    input_generator.file_path=./InputGenerator_custom.py \
    reinvent_common.prior_filename=./priors/formed.prior \
    reinvent_common.agent_filename=./priors/formed.prior \
    reinvent_common.max_steps=10 \
    diversity_scorer=All_weak_constraints
```

### Running All Diversity Strategies at Once

The provided script iterates over every configuration in `diversity_scorer/`, runs REINVENT4, and automatically applies t-SNE and diversity post-analysis:

```bash
cd $NAVIDIV_ROOT/Tutorials/Running_Reinvent/example/
chmod +x run_reinvent.sh
./run_reinvent.sh
```

### Output Structure

```
runs/test_case_2/
└── scaffold_only/
    ├── scaffold_only_1.csv          # Generated molecules + scores
    ├── scaffold_only_1_TSNE.csv     # With t-SNE coordinates
    ├── scorer_output/               # NaviDiv diversity analysis
    └── logs/                        # REINVENT4 logs
```

Load the resulting CSV in the NaviDiv app to visualize diversity evolution.

### Writing a Custom Diversity Configuration

Create a YAML file in `conf_folder/diversity_scorer/` to define your own constraints:

```yaml
# conf_folder/diversity_scorer/custom.yaml
scorer_dicts:
  - prop_dict:
      scorer_name: Scaffold
      scaffold_type: basic_wire_frame
      min_count_fragments: 1
      selection_criteria:
        diff_median_score: 0.05       # Stricter: require larger score improvement
    score_every: 5                    # Evaluate more frequently
    groupby_every: 10
    selection_criteria:
      count_perc_ratio: 2             # Stricter diversity ratio
      Total Number of Molecules with Substructure: 25
    custom_alert_name: customalertsscaffold

  - prop_dict:
      scorer_name: Fragments
      min_count_fragments: 5
      transformation_mode: none
    score_every: 15
    groupby_every: 30
    selection_criteria:
      count_perc_ratio: 10            # More relaxed fragment constraint
      Total Number of Molecules with Substructure: 100
    custom_alert_name: customalerts
```

**Key tuning parameters:**
- `count_perc_ratio` — lower = stricter diversity enforcement
- `Total Number of Molecules with Substructure` — cap on how many molecules may share a motif
- `score_every` — how often the diversity scorer runs (lower = more control, slower)
- `diff_median_score` — minimum score improvement required to accept a molecule

### Configuring the Target Property

Edit or create files in `conf_folder/stage_comp/` to change the optimization objective:

```yaml
# stage_comp/LogP.yaml  — optimize for drug-like LogP
model_params_rdkit_physchem:
  params_list: ["ALOGP"]
  lower_bound: [1.0]
  higher_bound: [3.0]
  weight: [1.0]
```

```yaml
# stage_comp/QED_MW.yaml  — multi-property optimization
model_params_rdkit_physchem:
  params_list: ["QED", "MW"]
  lower_bound: [0.5, 200.0]
  higher_bound: [1.0, 500.0]
  weight: [1.0, 0.5]
```

For additional scoring plugins, refer to the [REINVENT4 repository](https://github.com/MolecularAI/REINVENT4/tree/main) and the custom implementations in `src/navidiv/reinvent/reinvent_plugins/`.

### Post-run Analysis

```bash
# t-SNE projection (if not run automatically)
python3 $NAVIDIV_ROOT/src/navidiv/get_tsne.py \
    --df_path runs/test_case_2/scaffold_only/scaffold_only_1.csv \
    --step 20

# Comprehensive diversity analysis
python3 $NAVIDIV_ROOT/src/navidiv/run_all_scorers.py \
    --df_path runs/test_case_2/scaffold_only/scaffold_only_1_TSNE.csv \
    --output_path runs/test_case_2/scaffold_only/scorer_output
```

Then open the output CSV in the NaviDiv app for interactive exploration.

---

## Performance

- **Real-Time Analysis**: <3 seconds per 100 molecules on standard CPU
- **Scalable**: Complete analysis of 10,000 molecules in ~5 minutes
- **Memory Efficient**: Optimized for large-scale molecular datasets
- **Integration Ready**: Minimal computational overhead for existing workflows

## Citation

If you use NaviDiv in your research, please cite:

```bibtex
@article{azzouzi_navidiv:_2026,
	title = {{NaviDiv}: a web app for monitoring chemical diversity in generative molecular design},
	shorttitle = {{NaviDiv}},
	url = {https://pubs.rsc.org/en/content/articlelanding/2026/dd/d5dd00487j},
	doi = {10.1039/D5DD00487J},
	language = {en},
	urldate = {2026-04-16},
	journal = {Digital Discovery},
	author = {Azzouzi, Mohammed and Worakul, Thanapat and Corminboeuf, Clémence},
	year = {2026},
}
```


### Development Setup

```bash
git clone https://github.com/mohammedazzouzi15/NaviDiv.git
cd NaviDiv
pip install -e .[dev]
pre-commit install
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This work was supported by the Swiss National Science Foundation (SNSF).
