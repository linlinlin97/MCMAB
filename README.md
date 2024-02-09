# Multi-Task-CMAB-for-Budget-Allocation

This repository serves as supplementary material for our paper titled "Multi-Task Combinatorial Bandits for Budget Allocation".

## Abstract

Today’s top advertisers typically manage hundreds of campaigns simultaneously and consistently launch new ones throughout the year. A crucial challenge for marketing managers is determining the optimal allocation of limited budgets across various ad lines in each campaign to maximize cumulative returns, especially given the huge uncertainty in return outcomes. In this paper, we propose to formulate budget allocation as a multi-task combinatorial bandit problem and introduce a novel online budget allocation system. The proposed system

1. Integrates a Bayesian hierarchical model to intelligently utilize the metadata of campaigns and ad lines and budget size, ensuring efficient information sharing;
2. Provides modeling flexibility with linear mixed models, Gaussian Processes, and Neural Networks, catering to diverse environmental complexities;
3. Employs the Thompson sampling technique to strike a balance between exploration and exploitation. 

Through extensive empirical evaluations with both synthetic data and real-world data from Amazon campaigns, our system demonstrates robustness and adaptability, effectively maximizing the overall cumulative returns.

## Agents with Different Working Models

The main paper provides three choices for working models: Linear Regression (LR), Gaussian Process (GP), and Neural Network (NN).

#### LR Agents (`LR` folder)

 -  `_agent_LMM_MTB.py`: MCMAB with Linear Regression as the working model.
 -  `_agent_LMM_LB.py`: Feature-determined version with Linear Regression as the working model.
 -  `_agent_LMM_TS.py`: Feature-agnostic version with Linear Regression as the working model.

#### GP Agents (`GP` folder)

-  `_agent_GP_MTB_fast.py`: MCMAB with Gaussian Process as the working model.
-  `_agent_GP_fast.py`: Feature-determined version with Gaussian Process as the working model.

#### NN Agents (`NN` folder)

 -  `_agent_NeuralTS_MTB.py`: MCMAB with Neural Network as the working model.
 -  `_agent_NeuralTS.py`: Feature-determined version with Neural Network as the working model.

## Functions for Experiments

- `_env.py` and `real_env.py`: environment used for the simulation and the offline study of ADSP's campaign data.
- `_experiment.py`: Summarizes main functions used to run experiments in simulation.
- `experiment_LR.ipynb`, `experiment_GP.ipynb`, and `experiment_NN.ipynb`: Python notebooks for running sample experiments for each prior in {LR, GP, NN}.
- `_util.py`: helper functions.
- `_analyzer.py`: post-process simulation results.
- `Results_Plot.ipynb`: Python notebook summarizing simulation results.

## Others

- `Contextual_Bandit.py`: script of Han2021 adapted to the multi-task setting.
- `_optimizer.py`: script used for the optimization step.
