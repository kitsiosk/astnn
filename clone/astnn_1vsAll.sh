#!/bin/bash
#SBATCH --job-name="generalization-study"
#SBATCH --mail-user=konstantinos.kitsios@.uzh.ch
#SBATCH --mail-type=end,fail
#SBATCH --time=13:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gres gpu:1
#SBATCH --error=job-%j.err
#SBATCH --output=job-%j.out

module purge all
module load gpu
module load mamba

. /home/kkitsi/data/miniconda3/bin/activate
source activate cudasupport

#srun python pipeline_1vsAll.py --lang java
srun python train_1vsAll.py --lang java

# IMPORTANT:
# Run with                  sbatch run_ablation_slurm.sh
# check with                squeue --user=jn20t930 --jobs={job_id}
# monitor with              scontrol show --detail jobid {job_id}
# cancel with               scancel {job_id}
# monitor gpu usage with    ssh gnode14 and then nvidia-smi
# run interactive job with  srun --gres=gpu:rtx3090:1 --mem=64G --cpus-per-task=8 --time=24:00:00 --pty /bin/bash
