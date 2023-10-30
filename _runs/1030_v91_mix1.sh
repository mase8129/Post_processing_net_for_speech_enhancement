#!/bin/bash
#SBATCH -o ./jobs/1030_v91_mix1.%j.out            # Output-File
#SBATCH -J v91_mix1_22050	                                                # Job Name
#SBATCH --ntasks=1 			                                        # Anzahl Prozesse P 
#SBATCH --cpus-per-task=12       	                     	        # Anzahl CPU-Cores pro Prozess P
#SBATCH --gres=gpu:1                    	        	           # N GPUs anfordern
#SBATCH --time=7-00:00:00                         	                # estimated time
#SBATCH --partition=gpu						                        # which partition to run on
#SBATCH --mem=150G				                                    # XGiB resident memory pro node
#SBATCH --mail-type=ALL
#SBATCH --mail-user=marius.seipel@campus.tu-berlin.de

# add ld paths
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/beegfs/home/users/m/mase8129/scratch/marius-s/miniconda3/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/beegfs/home/users/m/mase8129/scratch/marius-s/miniconda3/envs/gpu-new/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/beegfs/home/users/m/mase8129/scratch/marius-s/miniconda3/envs/gpu-new/lib/python3.8/site-packages/tensorrt/

# load modules
module load nvidia/cuda/11.2 
module list

# check GPU allocation
nvidia-smi -L 
# run python script with args
python3 ./4_Model_train.py --n_epochs=30 --HPC=1 --model=91 --DS=5000 --loss_func=mix1 --batch_size=4 --data=voicefixer --sr=22050