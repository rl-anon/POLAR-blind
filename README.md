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

- Simulation studies  
- Real-data analysis  

Each component can be run separately.


## Simulation Experiments

To reproduce the simulation results:

1. Run the simulation script to generate action-value estimates:

```
bash code/simulation/simulation.py
```

2. After the simulation outputs are generated, reproduce the figures by running the notebook:

```
code/simulation/figs_simu.ipynb
```
Follow the instructions provided in each notebook cell to generate the figures reported in the manuscript.

## Real-Data Experiments

To reproduce the real-data results:

1. Run the real-data analysis script:
```
bash code/real_data/submit_real_data.sh
```

2. Generate the figures using python code/real_data/figs_real_data.py




