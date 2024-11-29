#!/bin/bash

#SBATCH --account r00213
#SBATCH -p general
#SBATCH --nodes=12
#SBATCH --ntasks-per-node=8
#SBATCH -o base_out_%j.out
#SBATCH -e base_error_%j.err
#SBATCH --mail-user=ls44@iu.edu
#SBATCH --mail-type=ALL
#SBATCH --time=2-30:00:00
#SBATCH --job-name=model_training

#Load any modules that your program needs
module load python
#pip install scikit-learn
#pip install tensorflow
#pip install contractions

echo "Modules loaded and required Python packages installed successfully."

python model.py --embedder_path ./output/gender_W2V_role_biasedEmbeddingsOut.w2v --csv_file ./data/Huang_et_al_2020/all_data_hash.tsv --emod biased_W2V
