#!/bin/bash

# Set arguments.
n_trials=5
n_hparams=500
account='NAISS2023-5-397'
gpu='T4'
container_path='/mimer/NOBACKUP/groups/inpole/singularity/inpole_env.sif'

# Check if the correct number of arguments were passed in.
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <parameters_file_path> <default_config_file_path>"
    exit 1
fi

wait_for_next_minute() {
    current_second=$(date +"%S")
    seconds_to_wait=$((60 - current_second))
    sleep $seconds_to_wait
}

# Get the input file paths.
parameters_file=$1
default_config_file=$2

# Create a temporary folder for the config files.
mkdir -p tmp_configs

# Activate the environment.
ml purge && ml Python/3.10.8-GCCcore-12.2.0 SciPy-bundle/2023.02-gfbf-2022b PyYAML/6.0-GCCcore-12.2.0
source sweep_env/bin/activate

i=0
while IFS=';' read -r state include_context_variables include_previous_treatment aggregate_history reduction add_current_context shift_periods estimators; do
    if [ $i -eq 0 ]; then
        # Skip the header row.
        i=$((i + 1))
        continue
    fi
  
    # Create a temporary config file.
    config_file="tmp_configs/config${i}.yml"
    cp "$default_config_file" "$config_file"

    # Update the config file.
    sed -ri "s/^(\s*)(include_context_variables\s*:\s*\S+\s*$)/\1include_context_variables: $include_context_variables/" "$config_file"
    sed -ri "s/^(\s*)(include_previous_treatment\s*:\s*\S+\s*$)/\1include_previous_treatment: $include_previous_treatment/" "$config_file"
    sed -ri "s/^(\s*)(aggregate_history\s*:\s*\S+\s*$)/\1aggregate_history: $aggregate_history/" "$config_file"
    sed -ri "s/^(\s*)(reduction\s*:\s*\S+\s*$)/\1reduction: $reduction/" "$config_file"
    sed -ri "s/^(\s*)(add_current_context\s*:\s*\S+\s*$)/\1add_current_context: $add_current_context/" "$config_file"
    sed -ri "s/^(\s*)(shift_periods\s*:\s*\S+\s*$)/\1shift_periods: $shift_periods/" "$config_file"

    wait_for_next_minute
    timestamp=$(date +"%y%m%d_%H%M")
    echo "$state $reduction $timestamp"

    estimators=$(echo "$estimators" | tr -d '\r')

    # Do not use quoting around $estimators, otherwise the script reads in all estimators as a single string.
    python scripts/run_experiment.py \
        --config_path "$config_file" \
        --estimators $estimators \
        --n_trials $n_trials \
        --n_hparams $n_hparams \
        --account "$account" \
        --gpu "$gpu" \
        --container_path "$container_path"

    i=$((i + 1))
done < "$parameters_file"
