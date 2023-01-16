# Introdution
This repo integrates [Human-Aware-RL](https://github.com/HumanCompatibleAI/human_aware_rl/tree/neurips2019) agent models with the [PantheonRL](https://github.com/Stanford-ILIAD/PantheonRL) framework for convenient human-ai coordination study on Overcooked. Changes are done under the [overcookedgym/overcooked-flask](https://github.com/LxzGordon/pecan_human_AI_coordination/tree/master/overcookedgym/overcooked-flask) directory.
<p align="center">
  <img src="./images/pecan_uni.gif" width="40%">
  <img src="./images/pecan_simple.gif" width="40%">
  <br>
</p>

# Instruction for usage

## 1. Install libraries
Install [PantheonRL](https://github.com/Stanford-ILIAD/PantheonRL) in this repo accordingly.
 ```shell
    pip install -e .
```

Install Overcooked in this repo
 ```shell
    pip install -e overcookedgym/human_aware_rl/overcooked_ai
```
## 2. Save models
Save models of [Human-Aware-RL](https://github.com/HumanCompatibleAI/human_aware_rl/tree/neurips2019) agents in this [directory](https://github.com/LxzGordon/pecan_human_AI_coordination/tree/master/models) (like the given MEP model for layout simple)
## 3. Start a process
For example, this will start a process on port 8008 with an MEP agent on the layout simple.
 ```shell
    python overcookedgym/overcooked-flask/app.py --layout=simple --algo=0 --port=8008 --seed=1
```

The next command will start a dummy demo agent.
 ```shell
    python overcookedgym/overcooked-flask/app.py --layout=simple --port=8008 --dummy=True
```
