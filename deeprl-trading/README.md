

# Deep Reinforcement Learning applied to Stock Trading


This repo contains the code for the paper "Deep Reinforcement Learning applied to Stock Trading" from Jan Komisarczyk, Younghoon Kim, and Mathias Ruoss.



## Requirements

To run our code, please install the following dependencies in a [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html) environment:


```
conda env create -f environment.yml 
```


---

## Reproduce our results from checkpoints:

The checkpoints of each trained agent-architecture combination are located in the ```checkpoints``` folder. Each model was trained with three different seeds. To reproduce the results in the paper, follow the below steps:

1) For a chosen agent-architecture combination, preferably copy all 3 seed checkpoints (```.pt``` files) to a separate folder in the project directory. 

2) Test each seed on the test period with the below command, where [...] indicates the options you have to set. The command saves the output in two ```.csv``` files to the specified ```--log_dir``` folder.
```
python main.py test --agent [a2c, ddpg or ppo] --checkpoint [path of .pt file] --log_dir [path to save output]
```

3) For each agent-architecture combination, repeat the above for each seed.

4) To average the seeds for a specific agent-architecture combination, preferably copy the generated ```"..."_returns.csv``` file from each seed to a separate folder and perform the below command:
```
python main.py average_seeds --checkpoint [path to folder with three ..._returns.csv files] --tag [custom tag]
```
This will save a new ```.csv``` file named after the provided ```--tag```, and containing daily averaged returns, to the checkpoint path under a new subfolder ```avg_returns```.

5) Repeat above for each agent-architecture combination.

6) Once we have the daily averaged returns for all six agent-architecture combinations, save again all ```.csv``` files to a folder. Then, create the plot and performance statistics with the below command:
```
python main.py performance --checkpoint [path to folder with all six .csv files in it]
```
*Note*: Make sure there is only **one** ```.csv``` file with the averaged daily returns for each agent-architecture combination in the ```--checkpoint``` folder, and not any other ```.csv``` file in it (it will assume any ```.csv``` file in it to be the daily returns of a trading strategy).

*Note2*: If no tag was provided in step 4, rename ```_returns.csv``` since to create a plot, no underscore can start the file.

7) Done! Enjoy a beautiful plot and performance statistics saved in a ```.csv```, a```.png``` and as LaTeX code in a ```.txt``` file in the automatically created folder ```results``` on the project directory level.

---
## Reproduce our results from scratch:
To reproduce our results from scratch run the following commands in your terminal with the environment activated:


1) For each agent-architecture combination, repeat the specified command three times with ```--seed 4```, -```-seed 5``` or ```--seed 6```.

*Note: Specify the checkpoint folder with an optional ```--tag``` argument for better overview. Also, training and validation is logged to tensorboard.*

- A2C CNN
```bash
python main.py train --agent a2c --seed [4, 5, 6] --train_iter 33000000 --ent_coef 1e-4
```
- A2C Transformer
```bash
python main.py train --agent a2c --seed [4, 5, 6] --train_iter 10000000 --arch transformer --ent_coef 1e-4
```
- DDPG CNN
```bash
python main.py train --agent ddpg --seed [4, 5, 6] --train_iter 4000000 --lr_actor 1e-5 --batch_size 128
```
- DDPG Transformer
```bash
python main.py train --agent ddpg --seed [4, 5, 6] --train_iter 3000000 --lr_actor 1e-5 --batch_size 128 --arch transformer
```
- PPO CNN
```bash
python main.py train --agent ppo --seed [4, 5, 6] --train_iter 10000000 --ent_coef 1e-1
```
- PPO Transformer
```bash
python main.py train --agent ppo --seed [4, 5, 6] --train_iter 10000000 --ent_coef 1e-1 --arch transformer
```

2) After training is done for each agent, find the automatically saved checkpoint files in the ```logs``` directory. The final model checkpoints should have been saved as  ```model.pt``` in each folder. Rename them to keep overview, e.g. ```"agent_arch_seed.pt"```.

3) Now we can test a model on the test period. Preferably copy the relevant and renamed model checkpoints to a new folder. Run the following commands with [...] specifying the options you have to set. This command saves the output to two ```.csv``` files to the specified ```log_dir```.
```
python main.py test --agent [a2c, ddpg or ppo] --checkpoint [path of .pt file] --log_dir [path to save output]
```

4) For each agent-architecture combination, repeat the above for each seed.

5) To average the seeds for a specific agent-architecture combination, preferably copy the generated ```"..."_returns.csv``` file from each seed to a separate folder and perform the below command:
```
python main.py average_seeds --checkpoint [path to folder with three ..._returns.csv files] --tag [custom tag]
```
This will save a new ```.csv``` file named after ```--tag```, and containing daily averaged returns, to the checkpoint path under a new subfolder ```avg_returns```.

6) Repeat above for each agent-architecture combination.

7) Once we have the daily averaged returns for all six agent-architecture combinations, save again all ```.csv``` files to a folder. Then, create the plot and performance statistics with the below command:

```
python main.py performance --checkpoint [path to folder with all six .csv files in it]
```

*Note*: Make sure there is only **one** ```.csv``` file with the averaged daily returns for each agent-architecture combination in the ```--checkpoint``` folder, and not any other ```.csv``` file in it (it will assume any ```.csv``` file in it to be the daily returns of a trading strategy).

*Note2*: If no tag was provided in step 5, rename ```_returns.csv``` since to create a plot, no underscore can start the file.

8) Done! Enjoy a beautiful plot and performance statistics saved in a ```.csv```, a```.png``` and as LaTeX code in a ```.txt``` file in the automatically created folder ```results``` on the project directory level.

#

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)