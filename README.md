# Introdution
This repo integrates [Human-Aware-RL](https://github.com/HumanCompatibleAI/human_aware_rl/tree/neurips2019) agent models with the [PantheonRL](https://github.com/Stanford-ILIAD/PantheonRL) framework for convenient human-ai coordination study on Overcooked. Changes are done under the [overcookedgym/overcooked-flask](https://github.com/LxzGordon/pecan_human_AI_coordination/tree/master/overcookedgym/overcooked-flask) directory.
<p align="center">
  <img src="./images/pecan_uni.gif" width="40%">
  <img src="./images/pecan_simple.gif" width="40%">
  <br>
</p>

# Instruction for usage

## 1. Install libraries
Install [PantheonRL](https://github.com/Stanford-ILIAD/PantheonRL) in this repo
 ```shell
    pip install -e .
```

Install [Overcooked](https://github.com/HumanCompatibleAI/human_aware_rl/tree/a1c70b2f44bbbdddc79b14678ceaf0367ee099f6) in this repo
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

#Citation
Please cite
'''shell
@article{Lou2023pecan,
  title={PECAN: Leveraging Policy Ensemble for Context-Aware Zero-Shot Human-AI Coordination},
  author={Lou, Xingzhou and Guo, Jiaxian and Zhang, Junge and Wang, Jun and Huang, Kaiqi and Du, Yali},
  journal={arXiv preprint arXiv:2301.06387},
  year={2023}
}
'''

'''shell
 @inproceedings{sarkar2022pantheonrl,
  title={PantheonRL: A MARL Library for Dynamic Training Interactions},
  author={Sarkar, Bidipta and Talati, Aditi and Shih, Andy and Sadigh, Dorsa},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={36},
  number={11},
  pages={13221--13223},
  year={2022}
}
'''

'''shell
@article{carroll2019utility,
  title={On the utility of learning about humans for human-ai coordination},
  author={Carroll, Micah and Shah, Rohin and Ho, Mark K and Griffiths, Tom and Seshia, Sanjit and Abbeel, Pieter and Dragan, Anca},
  journal={Advances in neural information processing systems},
  volume={32},
  year={2019}
}
'''
