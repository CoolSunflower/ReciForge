# ReciForge: Circuit Area and Optimal Recipe Prediction in Logic Synthesis

**Refer to `Preprint-RecipeOptimization` for detailed theoretical background.**

## Overview

ReciForge is a machine learning framework designed to predict the **Quality of Results (QoR)**—specifically, circuit area—after applying a sequence of synthesis commands (a “recipe”) to a circuit represented in the AIG format.

The framework introduces a **hybrid architecture** that integrates:

1. **Circuit Embedding Module** – Graph Neural Networks (GNNs) for structural representation of circuits.
2. **Recipe Processing Module** – Recurrent Neural Networks (RNNs) for modeling command sequences.
3. **Area Prediction Module** – A prediction component that combines circuit and recipe embeddings to estimate area at each synthesis step.

To improve accuracy and interpretability, ReciForge incorporates **domain-specific loss functions** emphasizing:

* Relative area reduction
* Critical optimization steps
* Sequential command dependencies
* Attention interpretability
* Feature consistency across circuit families

This enables not only **final area prediction** but also **step-by-step area progression tracking**, providing insights into the effectiveness of individual synthesis operations.

## Experimental Results

ReciForge was trained and evaluated on 8 benchmark designs:

* bc0
* apex1
* c6288
* c7552
* i2c
* max
* sasc
* simple_spi

Detailed results, training logs, and visualizations are available in the `ExperimentalResults` directory.

## Code Structure

* **main.py** – Main training script. Run as:

  ```bash
  python main.py {design-name}
  ```

  Requires `{design-name}.bench` in the `designs` folder and `{design-name}.csv` in the `datasets` folder.

* **loss.py** – Contains domain-dependent and domain-independent loss functions used in optimization.

* **visualiser.py** – Generates visualizations saved under `ExperimentalResults`.

* **inference.py** – Run inference for any recipe on a trained model:

  ```bash
  python inference.py {design-name} "rewrite" "balance" "refactor"
  ```

* **ExperimentalResults/** – Subdirectory per design with TensorBoard logs, training/testing plots, and logs.

* **weights/** – Stores the best fine-tuned model weights for each design.

* **finetuning/** – Includes fine-tuning scripts and usage instructions.

## Fine-Tuning

Refer to the dedicated `finetuning/README.md` for usage details.

## Optimal Recipe Prediction

ReciForge supports multiple strategies for optimal recipe generation: **Greedy Sampling, Monte Carlo Tree Search (MCTS), and Proximal Policy Optimization (PPO)**.

Usage:

```bash
python predict_recipe.py {design_name} --method {greedy|mcts|ppo} --length LENGTH \
    [--iterations ITERATIONS] [--episodes EPISODES] [--initial_area INITIAL_AREA] \
    [--output OUTPUT_FILE] [--compare]
```
