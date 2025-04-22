# QoR Area Prediction using Domain-Specific Loss Guided Graph Neural Networks

Done as part of ML for EDA Course Project.

## Problem Statement & Overview of Methodology
The goal is to predict the QoR (Area) of a circuit represented in an AIG format, after applying a given recipe.

We propose a hybrid architecture that combines Graph Neural Networks (GNNs) and Recurrent Neural Networks (RNNs) to address the circuit area prediction problem. Our approach leverages the representational power of GNNs to capture the structural properties of circuit designs and the sequential modeling capabilities of RNNs to process synthesis recipes.

The overall framework consists of three key components:
1. Circuit Embedding Module: A GNN-based component that transforms the circuit graph into a fixed-dimensional embedding
2. Recipe Processing Module: An RNN-based component that processes the sequence of synthesis commands
3. Area Prediction Module: A prediction component that combines circuit and recipe information to estimate area after each synthesis step

Additionally, we introduce domain-specific loss functions that incorporate circuit optimization knowledge to guide the learning process and improve prediction accuracy. These specialized loss functions emphasize relative area reduction, critical optimization steps, attention interpretability, and meaningful circuit representations.

The proposed approach not only predicts the final area after applying the complete recipe but also provides a step-by-step prediction of area changes throughout the optimization process, offering valuable insights into the effectiveness of each synthesis command.

## Experimental Results
The model was trained for 8 designs in total:
1. bc0
2. apex1
3. c6288
4. c7552
5. i2c
6. max
7. sasc
8. simple_spi

The detailed experimental results, training logs and test visualisations are avalaible in `ExperimentalResults` folder.

## Code Structure

main.py: Main training script. To run `python main.py {design-name}`. Ensure that {design-name}.bench file is located in the designs folder and {design-name}.csv is located in the datasets folder with relevant random simulations in the format similar to already present file.

loss.py: Utility file used by main.py. It contains the main loss functions (both domain dependent and domain indepdent) that was used in the optimisation process.

visualiser.py: Utility file used by main.py. It contains code for plotting the various graphs saved in the ExperimentalResults folder.

inference.py: Run inference for any recipe and any design (assuming trained model for that design is located in the weights folder). Run `python inference.py {design-name} "rewrite" "balance" "refactor"`

ExperimentalResults/*: Contains subfolder for each design, each of which contains the tensorboard training logs, training visualisations, testing visualisations, and training logs.

weights/*: Contains best weights for each of the fine-tuned designs.

finetuning/*: Contains finetuning scripts and command information


## Finetuning

Refer to the README file in finetuning folder

## Optimal Recipe Prediction

Usage:
    python predict_recipe.py {design_name} --method {greedy|mcts|ppo} --length LENGTH 
        [--iterations ITERATIONS] [--episodes EPISODES] [--initial_area INITIAL_AREA] 
        [--output OUTPUT_FILE] [--compare]