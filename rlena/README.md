# Dev Branch
When merging to master, bypass dev branch first!


# RLENA: Battle Arena for RL agents
## RL2021 project

codes for envs included as submodules for convenience.  
**DO NOT MODIFY AND COMMIT CODES IN ENVS/\<env>**  
this may removed later.

python version == 3.7.9  

```bash
git clone --recursive <project url>
```

```bash
cd envs/marlenv
pip install -e .

cd ../envs/playground
pip install -U .
```

rl2 library included as a submodule for convenience.  
Currently it is mirroring the folked version in diya gitlab.  
This also may removed later.  
if you need to install rl2 for running the code,
```bash
cd misc/rl2
pip install -e .
```
  
marlenv: Snake   
playground: Pommerman

## implement RL algorithms
1. DQN
2. RAINBOW
3. PPO
4. QMIX
5. COMA
6. SAC

## demo
Need more ideation

## Execute command
> QMIX : python main.py --algo=QMIX --config_dir=config/qmix/QMIX.yaml
