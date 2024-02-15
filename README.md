# Fair Machine Unlearning: Data Removal while Mitigating Disparities
---
### [Paper](https://arxiv.org/abs/2307.14754)

The official code to replicate the results found in *Fair Machine Unlearning: Data Removal while Mitigating Disparities*

## Usage

To unlearn at random, execute the following script. `--protected-attribute` determines what feature to measure fairness metrics over, and `--std` deterrmines the noise added to make the method unlearnable.

```
python unlearning_disparity_experiments.py --dataset "COMPAS" --protected-attribute "race" --reg 1e-4 --std 1e0
```

To unlearn from a specific subgroup, include the following flags specifying `--unlearned-attribute` and `--value`:

```
python unlearning_disparity_experiments.py --dataset "Adult" --protected-attribute "race" --reg 1e-4 --std 1e0 --unlearned-attribute "race" --value 0
```

Finally, to test on fair or unfair loss functions, include the `--fair` flag and a corresponding fairness regularizer value:

```
python unlearning_disparity_experiments.py --dataset "Adult" --protected-attribute "race" --reg 1e-4 --std 1e0 --unlearned-attribute "race" --value 0 --fair --fairreg 1e0 
```

`eps_delta_experiments.py` computes accuracy while sweeping over pairs of $\epsilon, \delta$ to construct unlearning trade-off curves.

`fairness_tradeoff_experiments.py` computes accuracy while sweeping over $\gamma$ to construct fairness trade-off curves.

## API

The main functions for fair unlearning, and replication of previous methods are all in `fair_retraining.py`. The other scripts are infrastructure to run and replicate experiments.s