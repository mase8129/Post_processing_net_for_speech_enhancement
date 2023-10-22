#!/bin/bash
#SBATCH -o ./jobs/%j.out            # Output-File
#SBATCH -J HiFi_v811	                                                # Job Name
#SBATCH --ntasks=1 			                                        # Anzahl Prozesse P 
#SBATCH --cpus-per-task=12       	                     	        # Anzahl CPU-Cores pro Prozess P
#SBATCH --gres=gpu:1                    	        	           # N GPUs anfordern
#SBATCH --time=50:00:00                         	                # estimated time
#SBATCH --partition=gpu						                        # which partition to run on
#SBATCH --mem=100G				                                    # XGiB resident memory pro node
#SBATCH --mail-type=ALL
#SBATCH --mail-user=marius.seipel@campus.tu-berlin.de

# load modules
module load nvidia/cuda/11.2 
module list

# check GPU allocation
nvidia-smi -L 
# run python script with args
python3 ./4_Model_train.py --n_epochs=100 --HPC=1 --model=811 --DS=2000