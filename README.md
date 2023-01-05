# Introdution
This repo integrates [Human-Aware-RL](https://github.com/HumanCompatibleAI/human_aware_rl/tree/neurips2019) agent models with the [PantheonRL](https://github.com/Stanford-ILIAD/PantheonRL) framework for convenient human-ai coordination study. Changes are done under the [overcookedgym/overcooked-flask](https://github.com/LxzGordon/pecan_human_AI_coordination/tree/master/overcookedgym/overcooked-flask) directory.


# Instruction
Save the models from (Human-Aware-RL)[https://github.com/HumanCompatibleAI/human_aware_rl/tree/neurips2019] agent models in this (directory)[https://github.com/LxzGordon/pecan_human_AI_coordination/tree/master/models] (like the given MEP model for layout simple)

For example, this will start a process on port 8008 with an MEP agent on the layout simple. For dummy agent in the demo layout, set dummy=True
 ```shell
    cd overcookedgym/overcooked-flask
    python app.py --layout=simple --algo=0 --port=8008 --seed=1 --dummy=False
```
