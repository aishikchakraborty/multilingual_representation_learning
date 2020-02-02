#!/bin/bash
#SBATCH --account=rpp-bengioy
#SBATCH --ntasks=1
#SBATCH --mem=40000M
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chakraba@mila.quebec
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --output=outputs/logs/multibert_translate_nonfinetuned.out
#SBATCH --error=outputs/logs/multibert_translate_nonfinetuned.out


python -u main.py --cuda --src en --tgt de --epochs 0
