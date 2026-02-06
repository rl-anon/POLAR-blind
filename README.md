# POLAR 
This is a Python implementation of *POLAR: A Pessimistic Model-based Policy Learning Algorithm for Dynamic Treatment Regimes*

## Abstract: 
Dynamic treatment regimes (DTRs) provide a principled framework for optimizing sequential decision-making in domains where decisions must adapt over time in response to individual trajectories, such as healthcare, education, and digital interventions. However, existing statistical methods often rely on strong positivity assumptions and lack robustness under partial data coverage, while offline reinforcement learning approaches typically focus on average training performance, lack statistical guarantees, and require solving complex optimization problems. To address these challenges, we propose POLAR, a novel pessimistic model-based policy learning algorithm for offline DTR optimization. POLAR estimates the transition dynamics from offline data using parametric or nonparametric methods and quantifies uncertainty for each history-action pair. A pessimistic penalty is then incorporated into the reward function to discourage actions associated with high uncertainty. Unlike many existing methods that focus on average training performance, POLAR directly targets the suboptimality of the final learned policy and offers theoretical guarantees, without relying on computationally intensive minimax or constrained optimization procedures. To the best of our knowledge, POLAR is the first model-based DTR method to provide both statistical and computational guarantees, including finite-sample bounds on policy suboptimality. Empirical results on both synthetic data and the MIMIC-III dataset demonstrate that POLAR outperforms existing methods in terms of policy value and produces reliable, history-aware treatment strategies.

# Getting Started
Install the dependencies using:

```
pip install -r requirements.txt
```

# Run Instructions
To run the POLAR simulation, execute the following shell script:

```
bash simulation/submit_simulation.sh
```


To run the POLAR real data experiment, execute the following shell script:

```
bash real_data/submit_real_data.sh
```
These scripts will initiate the full pipeline for simulating environments, estimating transition models, and training the POLAR policy.


# Reproducibility

This repository provides code to reproduce both the simulation studies and the real-data analyses presented in the manuscript.  
The experiments are organized into two independent components:

- **Simulation studies**
- **Real-data analyses**

Each component can be run independently after installing the required dependencies:

```bash
pip install -r requirements.txt
```

## Simulation Experiments

The simulation data are fully self-contained and are generated directly by the provided code.

To reproduce the simulation results:

1. Run the simulation pipeline:

    ```
    bash code/simulation/submit_simulation.sh
    ```
This script performs environment simulation, transition model estimation, and POLAR policy training.

2. After the simulation outputs are generated, reproduce the figures using:

    ```
    code/simulation/figs_simu.ipynb
    ```

Follow the instructions in each notebook cell to generate the figures reported in the manuscript.

## Real-Data Experiments

The real-data analysis is based on the publicly available MIMIC-III critical care database.
Access to MIMIC-III requires credentialed approval via PhysioNet:

👉 https://physionet.org/content/mimiciii/1.4/

After obtaining access, the processed dataset used in this study
(sepsis_processed_state_action.csv) can be reproduced using publicly available preprocessing pipelines:

https://github.com/matthieukomorowski/AI_Clinician

https://github.com/microsoft/mimic_sepsis

Additional feature-description files (CSV and TXT) are included in this repository to document the extracted variables.

To reproduce the real-data results:

1. Run the real-data analysis pipeline:

    ```
    bash code/real_data/submit_real_data.sh
    ```
This script performs dataset processing, model training, and policy evaluation.

2. Generate the figures using: real_data/figs_real_data.py

