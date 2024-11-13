#!/usr/bin/env bash
#SBATCH -A NAISS2024-5-480
#SBATCH -p alvis
#SBATCH -N 1
#SBATCH --gpus-per-node=A40:1
#SBATCH -t 1-0:0  # days-hours:minutes

container_path='/mimer/NOBACKUP/groups/inpole/singularity/inpole_env.sif'

# Check if an argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 experiment"
  exit 1
fi

experiment=$1

cd ~
rsync -r inpole $TMPDIR --exclude='*_env'
cd $TMPDIR/inpole

if [ "$experiment" == "sepsis" ]; then
    echo "Evaluating sepsis models..."
    apptainer exec --bind $HOME:/mnt --nv $container_path python scripts/perform_model_analysis.py \
        --experiment sepsis \
        --data_path /mimer/NOBACKUP/groups/inpole/data/sepsis_data.pkl \
        --out_path /mimer/NOBACKUP/groups/inpole/results
elif [ "$experiment" == "ra" ]; then
    echo "Evaluating RA models..."
    apptainer exec --bind $HOME:/mnt --nv $container_path python scripts/perform_model_analysis.py \
        --experiment ra \
        --data_path /mimer/NOBACKUP/groups/inpole/data/ra_Xgy_incl_cdai.pkl \
        --out_path /mimer/NOBACKUP/groups/inpole/results
else
    echo "Invalid experiment type. Please choose 'sepsis' or 'ra'."
    exit 1
fi
