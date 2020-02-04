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
#SBATCH --output=outputs/logs/multibert_translate_finetuned.out
#SBATCH --error=outputs/logs/multibert_translate_finetuned.out

SRC=en
TGT=(
    "ar"
    "bg"
    "de"
    "el"
    "es"
    "fr"
    "hi"
    "ru"
    "th"
    "vi"
    "zh"
)

for tgt in "${TGT[@]}"
do
   python preprocess.py --src $SRC --tgt $tgt
    echo "Done preprocessing for $SRC-$tgt"
done

for tgt in "${TGT[@]}"
do
   python main.py --cuda --epochs 1 --src $SRC --tgt $tgt
done
