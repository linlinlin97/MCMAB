# Multi-Task-CMAB-for-Budget-Allocation

This repository serves as supplementary material for our paper titled "Multi-Task Combinatorial Bandits for Budget Allocation".

## Abstract

Today’s top advertisers typically manage hundreds of campaigns simultaneously and consistently launch new ones throughout the year. A crucial challenge for marketing managers is determining the optimal allocation of limited budgets across various ad lines in each campaign to maximize cumulative returns, especially given the huge uncertainty in return outcomes. In this paper, we propose to formulate budget allocation as a multi-task combinatorial bandit problem and introduce a novel online budget allocation system. The proposed system

1. Integrates a Bayesian hierarchical model to intelligently utilize the metadata of campaigns and ad lines and budget size, ensuring efficient information sharing;
2. Provides modeling flexibility with linear mixed models, Gaussian Processes, and Neural Networks, catering to diverse environmental complexities;
3. Employs the Thompson sampling technique to strike a balance between exploration and exploitation. 

Through extensive empirical evaluations with both synthetic data and real-world data from Amazon campaigns, our system demonstrates robustness and adaptability, effectively maximizing the overall cumulative returns.

## Agents with Different Working Models

The main paper provides three choices for prior distribution: Linear Regression (LR), Gaussian Process (GP), and Neural Network (NN).

#### LR Agents (`LR` folder)

 -  `_agent_LR_MTB.py`: Posterior updating procedure for LR in MTB (concurrent) version.
 -  `_agent_LR_LB.py`: Posterior updating procedure for LR in sequential version.
 -  `_env_LR.py`: Simulation environment for LR used to generate results in the paper.

#### GP Agents (`GPTS` folder)

-  `_agent_GP_MTB_concurrent_fast.py`: Posterior updating procedure for GP in MTB (concurrent) version, with batch update and memory check to speed up posterior calculation.
-  `_agent_GP_concurrent_fast.py`: Posterior updating procedure for GP in sequential version, with batch update and memory check to speed up posterior calculation.
-  `_env_GP.py`: Simulation environment for GP used to generate results in the paper.

#### NN Agents (`NN` folder)

 -  `_agent_NeuralTS_MTB.py`: Posterior updating procedure for NN in MTB (concurrent) version.
 -  `_agent_NeuralTS.py`: Posterior updating procedure for NN in sequential version.
 -  `_env_NN.py`: Simulation environment for NN used to generate results in the paper.

## Experiment Scripts

- `_experiment.py`: Summarizes main functions used to run experiments in simulation.
- `experiment_LR.ipynb`, `experiment_GP.ipynb`, and `experiment_NN.ipynb`: Python notebooks for running sample experiments for each prior in {LR, GP, NN}.
- `experiment_real.ipynb`: Python notebook for running real data from Amazon DSP side.

## Others

- `Simulation` folder: Provides additional Python scripts to generate results in the simulation section of our main paper.
- `optimization` folder: Holds supplementary testing files for the optimization step in our main algorithm.

Feel free to explore the provided sample notebooks in the main repository.

## Contact

Please contact {lge, yxu63, jchu3}@ncsu.edu if you encounter any issues when running the code.
