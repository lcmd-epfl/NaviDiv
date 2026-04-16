#!/bin/bash

# Check if conda is available
if ! command -v conda &> /dev/null
then
    echo "conda could not be found. Please install Anaconda or Miniconda."
    exit 1
fi

# Automatically detect script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Navigate up to find NaviDiv root (3 levels up from example/ folder)
NAVIDIV_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Environment and path configuration
ENV_NAME="reinvent4"
export PYTHONPATH="${PYTHONPATH}:${NAVIDIV_ROOT}/src/navidiv/reinvent/"
python_script_path="${NAVIDIV_ROOT}/src/navidiv/reinvent/"
config_path="${SCRIPT_DIR}/conf_folder"
config_name=test
wd="${SCRIPT_DIR}/runs/test_case"
i=2

# Create output directory
wd_current="${wd}_${i}/"

# Run all diversity scorer configurations
for diversity_config in "$config_path"/diversity_scorer/*.yaml; do
    diversity_scorer_name=$(basename "$diversity_config" .yaml)
    run_name=$diversity_scorer_name
    
    mkdir -p "$wd_current"
    echo "Running with diversity scorer: $diversity_scorer_name, run $i"
    
    # Run REINVENT with specific diversity scorer
    conda run -n $ENV_NAME python3 "$python_script_path/run_reinvent_2.py" \
        --config-name $config_name \
        --config-path $config_path \
        name=$run_name \
        wd=$wd_current \
        input_generator.file_path="${SCRIPT_DIR}/InputGenerator_custom.py" \
        reinvent_common.prior_filename="${SCRIPT_DIR}/priors/formed.prior" \
        reinvent_common.agent_filename="${SCRIPT_DIR}/priors/formed.prior" \
        reinvent_common.max_steps=100 \
        diversity_scorer=$diversity_scorer_name

    # Post-processing: t-SNE visualization
    conda run -n $ENV_NAME python3 "${NAVIDIV_ROOT}/src/navidiv/get_tsne.py" \
        --df_path "$wd_current/${run_name}/${run_name}_1.csv" \
        --step 20

    # Post-processing: Diversity analysis
    conda run -n $ENV_NAME python3 "${NAVIDIV_ROOT}/src/navidiv/run_all_scorers.py" \
        --df_path "$wd_current/${run_name}/${run_name}_1_TSNE.csv" \
        --output_path "$wd_current/${run_name}/scorer_output"
done