# MLOps Assignment 1

## Setup
1. Create and activate a conda environment:
   ```bash
   conda create -n mlops python=3.9 -y
   conda activate mlops

2. pip install -r requirements.txt
python train.py# MLOps Assignment 1

##  Overview
This repository contains the implementation for **MLOps Assignment 1**.  
We train and evaluate two regression models on the **Boston Housing dataset**:
- Decision Tree Regressor
- Kernel Ridge Regressor

The work is structured across three branches:
- `main` → base branch with README and merged code
- `dtree` → contains `requirements.txt`, `misc.py`, and `train.py` (Decision Tree)
- `kernelridge` → contains `train2.py` (Kernel Ridge) and GitHub Actions workflow

---

## Setup Instructions

1. **Create and activate conda environment**
   ```bash
   conda create -n mlops python=3.9 -y
   conda activate mlops
2. Install dependencies
   pip install -r requirements.txt

3. Running the Models
Decision Tree (train.py)   
python train.py

Kernel Ridge (train2.py)
python train2.py
