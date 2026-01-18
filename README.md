# SAFE-MolGen

[![Python Version](https://img.shields.io/badge/python-3.11.7-blue)](https://www.python.org/)
[![Anaconda Version](https://img.shields.io/badge/anaconda-2.5.2-green)](https://anaconda.org/)
[![License](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

This repository contains the datasets and codes for **_SAFE-MolGen_**. We also deployed website version for this work with the link of <a href='https://www.safe.lanl.gov/molgen/doc' target='_blank'>www.safe.lanl.gov/molgen/doc</a>. 

Below are the details for this repository.  

## Repository Folder Structure Description

The development of the method is divided into several steps, each represented by a folder: 
- Prepare the datasets (`0_prepData`). 
- Train supervised ML models (`s1_trainML`). 
- Run SAFE-MolGen (`s2_runfSepaMolGen`).
- Postprocess the results of SAFE-MolGen (`s3_postprocResults`).
- Website service guide and NERSC SuperFacility API (`s4_web`).
- View all GOOD and NEW extractant molecules (`s5_viewAllMols`).

## Usage

To get started, it is recommended to first create a `Python 3.11.7` virtual environment using Anaconda. Then, clone the repository and install the required packages listed in `requirements.txt`. Below is the code to start:  
```bash
git clone https://github.com/BaosenZ/SAFE-MolGen-LLM.git
conda create -n env-safemolgen python=3.11.7 -y
conda activate env-safemolgen
cd to/SAFE-MolGen-LLM
pip install -r requirements.txt
conda install openbabel==3.1.1.1
```
Add `.env` file in the `SAFE-MolGen-LLM` folder with valid OpenAI API key. After that, navigate to the desired folder. Follow the instructions in the relevant script files within each folder to run the code. 

## Contributing

Feel free to fork this repository and/or submit issues for any bugs.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## How to Cite

Zhang, B.; Summers, T. J.; Augustine, L. J.; Taylor, M. G.; Geist, A.; Li, R.; Batista, E. R.; Perez, D.; Yang, P.; Schrier, J. Augmenting Large Language Models for Automated Discovery of f-Element Extractants. J. Am. Chem. Soc. 2026. DOI: https://doi.org/10.1021/jacs.5c19738.