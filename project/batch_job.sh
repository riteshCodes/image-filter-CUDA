#!/bin/bash
#SBATCH -J Cuda-Lab
#SBATCH -e /home/kurse/kurs00042/rs83muwy/Praktikum_CUDA/Lab/log.err.%j
#SBATCH -o /home/kurse/kurs00042/rs83muwy/Praktikum_CUDA/Lab/log.out.%j
#SBATCH -n 1                  # Prozesse
#SBATCH -c 1                  # Kerne pro Prozess
#SBATCH --mem-per-cpu=1600    # Hauptspeicher in MByte pro Rechenkern
#SBATCH -t 00:02:00           # in Stunden und Minuten, oder '#SBATCH -t 10' - nur Minuten
#SBATCH --exclusive
#SBATCH -C "nvd"
#SBATCH --account=kurs00042
#SBATCH --partition=kurs00042
#SBATCH --reservation=kurs00042
#SBATCH --mail-type=ALL
# -------------------------------
module load cuda
cd /home/kurse/kurs00042/rs83muwy/Praktikum_CUDA/project_code
make
./bilateral test_input.ppm 5
