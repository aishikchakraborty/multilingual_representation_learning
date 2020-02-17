#!/bin/bash
#SBATCH --account=rpp-bengioy
#SBATCH --ntasks=1
#SBATCH --mem=40000M
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chakraba@mila.quebec
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --output=../outputs/logs/en_extract.out
#SBATCH --error=../outputs/logs/en_extract.out

./extract_and_clean_wiki_dumps.sh /scratch/aishikc/enwiki-latest-pages-articles-multistream.xml.bz2

