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
#SBATCH --job-name=debias_embedding


#Load any modules that your program needs
module load python
pip install scipy
pip install gensim
pip install torch
pip install scikit-learn

echo "Modules loaded and required Python packages installed successfully."

python debias.py "W2V" 'data/vocab/gender_attributes_optm.json' role -v -hard -k 3 -w 