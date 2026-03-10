# MORL Extension of PPO for L2RPN (Grid2Op)

## Background
This repository extends the PPO solution for the NeurIPS 2020 L2RPN competition with:

* Stage-based orchestration
* Multi-Objective Reinforcement Learning (MORL)
* Gated and preference-conditioned scalarization
* Config-driven experiment control
* Structured checkpoint deployment
* WandB-based evaluation

The PPO backbone originates from:
[Aspirin96 – L2RPN_NIPS_2020_a_PPO_Solution](https://github.com/Aspirin96/L2RPN_NIPS_2020_a_PPO_Solution)

This repository preserves the original training pipeline and augments it with MORL objectives and orchestration logic.

## Summary
The L2RPN competition calls for solutions of power grid dispatch based on Reinforcement Learning (RL). According to the game rules, three categories of actions can be implemented to realize the power grid dispatch: **substation switch**, **line switch**, and **generator production**. The first two control the grid topology, while the last one controls the grid injection.  
For "line switch" action, it is always beneficial to reconnect a transmission line in most cases. Thus, "line switch" action is determined by the expert experience in our solution.  
For "generator production" action, we find its costs are far more than the payoffs brought in almost all cases, since the production modification to the power contract should be compensated. **It is not an economic dispatch problem!** Thus, production redispatch is excluded in our solution.  
That is to say, our RL agent considers topology control only. Even so, there are still two challenges to obtain a well-performed agent:

* Humongous action space  
The action space involves the combination explosion issue. In this competition, there are more than 60k legal "substation switch" actions. It is almost impossible for the RL agent to learn knowledge from such a huge action space.
* Strongly constrained game  
To some extent, a power system is a fragile system: any mistake in dispatch may lead to a cascading failure (**blackout**), which means a game over in the competition. It decides the RL agent is hard to learn from the early random exploration, because it probably finds almost all actions lead to a negative reward!  

To deal with the above two issues, we propose a "*Teacher-Tutor-Junior Student-Senior Student*" framework in our solution.

![illustration](./img/illustration.png)

+ Teacher: action space generation  
*Teacher* is a strategy that finds a greedy action to minimize the maximum load rate (power flow/capacity) of all lines through enumurating all possible actions (~60k). Obviously, it is a time-consuming expert strategy.  
By calling *Teacher* Strategy in thousands of scenarios, we obtain an action library consisting of actions chosen in different scenarios. In the following procedure, we treat this action library as the action space.  
In our final solution, we filter out some actions that occur less frequently, and the size of final action library is 208. It well solves the issue of **humongous action space**.
+ Tutor: expert agent  
*Tutor* is also a greedy strategy similar to *Teacher*, the difference is that its search space is the reduced action space (208 in our solution), rather than the original action space. As a result, *Tutor* has a far higher decision-making speed. **In this competition, the *Tutor* strategy can achieve a score of 40~45.**
+ Junior Student: imitation learning  
To address the second challenge, we pre-train a neural network whose input is observation and output is probability distribution of all actions (i.e., *Actor*), to imitate the expert agent *Tutor*. We call this agent as *Junior Student*.  
Specifically, we feed *Tutor* different observations, and obtain corresponding greedy actions. Then, construct the labels by setting the probability of the chosen action as 1 while others as 0. Finally, train the neural network in a supervised learning manner, with the dataset in the form of (*feature*: observation, *label*: action probability).  
In our solution, the top-1 accuracy of *Junior Student* reaches 30%, while top-20 accuracy reaches 90%. **The *Junior Student* strategy can achieve a score of ~40 in the competition.**
+ Senior Student: reinforcement learning  
To achieve a better performance than *Tutor*, we build a **PPO** model named *Senior Student*, whose **Actor Network** is copied from *Junior Student*.  
Different from *Tutor* considers the short-term reward only, *Senior Student* focuses on the long-term reward and therefore performs better in the competition. **The *Senior Student* strategy can achieve a score of ~50**.  
It is worth noting that ***Senior Student* faces a risk of overfitting**.

See more technical details in this [video](https://drive.google.com/file/d/1dnq-QgpAVMpWuSQRZPrVzunIkaae7SGW/preview) (released, click to see).

## New Components
* orchestrate_training.py
Stage-based wrapper controlling the full pipeline.
* config_orchestrator.json
Central configuration file (stages, MORL weights, deployment).
* morl_objectives.py
Implements:
  * Dataset-driven metadata construction
	* Fairness, sustainability, structural metrics
	* Gated and preference-conditioned scalarization
* analyze_morl_wandb_runs.py
Post-training metric analysis and trade-off inspection.
* Senior Student
Multiple Senior student scripts have been added in order to be able to get baseline readings and test the MORL implementations

## MORL Design
The reward is decomposed into blocks:
Block	Purpose
Primary	Longevity / survival
Fairness	Line load variance, curtailment equity
Sustainability	CO₂ emissions, renewable ratio
Structural	Risk proxy, N-1 proxy, economic cost, simplicity, L2RPN score
Two scalarization modes are supported:
* Gated MORL
Secondary objectives activate only after survival exceeds a threshold.
* Preference-Conditioned MORL
Reward = weighted combination of objective blocks.
Weights are configured in config_orchestrator.json.

## Setup

### 1. Install Git LFS (required for model weights)

Skip this part if you plan to retrain the models from scratch.

```
brew install git-lfs # MacOS
git lfs install. # Windows
git lfs pull
```

### 2. Install Python dependencies

```
pip install -r requirements.txt
```


## Reproducibility

The full pipeline is controlled via `config_orchestrator.json` and executed through `orchestrate_training.py`. No manual stage execution is required.

Before running long experiments, you must manually set the wall-clock time limit in:

- `orchestrate_training.py`
- The `SeniorStudent*.py` script you plan to run  

This guarantees a smooth and controlled shutdown.

---

* step 1. Configure the pipeline  

  Edit `config_orchestrator.json` to select which stages to execute and which SeniorStudent variant to use.

  Example:

  ```json
  "stages": {
    "run_teacher1": false,
    "run_teacher2": false,
    "generate_action_space": false,
    "generate_tutor_dataset": false,
    "junior_train": false,
    "junior_convert": false,
    "senior_train": true,
    "gated_tiered_morl": true,
    "pereference_condition": false,
    "do_nothing": false,
    "random_action": false,
    "deploy_checkpoint": true,
    "run_runner": true
  }
  ```

  Stage selection works as follows:

  - `run_teacher*` → generates action space  
  - `generate_tutor_dataset` → runs Tutor  
  - `junior_train` / `junior_convert` → imitation learning  
  - `senior_train` → activates RL training  
  - `gated_tiered_morl` → runs `SeniorStudentMORL.py`  
  - `pereference_condition` → runs `SeniorStudentPrefConMORL.py`  
  - `do_nothing` → baseline  
  - `random_action` → baseline  
  - `deploy_checkpoint` → copies best PPO model to `./submission/ppo-ckpt/`  
  - `run_runner` → executes `runner.py` after training  

  Only one SeniorStudent mode should be active at a time.

  MORL weights and scalarization parameters are configured in the `"morl"` section of the same file.

---

* step 2. Run the full pipeline  

  From repository root:

  ```
  python orchestrate_training.py
  ```

  The orchestrator:

  - Executes enabled stages in sequence  
  - Handles Teacher parallelization  
  - Trains Junior and Senior modules  
  - Enforces global runtime limit  
  - Deploys checkpoint automatically (if enabled)  
  - Optionally runs evaluation  

---

* step 3. Adjust wall-clock time limit  

  Before long runs, modify:

  - `MAX_RUNTIME_SECONDS` in `orchestrate_training.py`
  - The corresponding time limit in the selected `SeniorStudent*.py` script  

  These must be some time (we used 30 min) shorter then your compute budget (e.g., cluster job time limit).  
  The pipeline relies on this in order to terminate cleanly before the time limit runs out and forces a termination.

---

* step 4. Configure MORL objective parameters  

  Objective weights and scalarization behavior are defined in the `"morl"` section of `config_orchestrator.json`.

  Example:

  ```json
  "morl": {
    "w_fair_rho": 0.0,
    "w_fair_curt": 0.0,
    "w_equity": 0.0,
    "w_ren": 0.0,
    "w_co2": 0.0,
    "w_risk": 0.0,
    "w_n1": 0.0,
    "w_econ": 0.0,
    "w_simplicity": 1.0,
    "w_l2rpn": 0.0,
    "tau_primary": 0.5,
    "alpha_fair": 0.1,
    "alpha_sust": 0.1,
    "alpha_struct": 0.3
  }
  ```

  Parameters control:

  - Individual objective weights  
  - Primary survival gating threshold (`tau_primary`)  
  - Contribution scaling of fairness, sustainability, and structural blocks  

  These directly affect the scalar reward construction used during training.


## Extra Tips
+ Tutorials for Grid2op  
Grid2op environment is somehow complex, some tutorials are provided by [@Benjamin D.](https://github.com/BDonnot) in the form of Jupyter Notebooks [here](https://github.com/rte-france/Grid2Op/tree/master/getting_started).
+ Acclerate Power Flow Calculation  
Noticed the main computational burden lies in "power flow calculation", it is better to install [Lightsim2grid](https://github.com/BDonnot/lightsim2grid) which is ~10 times faster than default pandapower backend. It helps accelerate training effectively.
+ Improve Neural Network Implementation  
In our solution, the neural network is implemented naively: we just send the original observation to a full-connected network. More advanced skills such as **graph convolution network** and **attention mechanism** are recommended to improve the performance.
+ Improve Sampling Efficiency  
Noticed the sampling time is much longer than the updating time of neural networks, soft actor-critic (SAC) which is **off-policy** can be trained to replace the proximal policy optimization (PPO) in the current solution. It will improve sampling efficiency significantly, which further accelerates training.

