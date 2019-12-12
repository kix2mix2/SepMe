# SepMe

This repository contain the Python implementation for the research available at this site: http://sepme.cs.univie.ac.at/index.php



Running Instructions:

0. Create a virtual env from `conda.yml`: `conda env create -f conda.yml`
1. `pip install . -e`
2. `mlflow server`
3. `python -m SepMe.main`.x



To update conda file: `conda env export | grep -v "^prefix: " > conda.yml`
