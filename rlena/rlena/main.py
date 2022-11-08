# Entrypoint
import argparse
import random
import numpy as np
import torch
import os

from rlena.algos import runners


def main():
    parser = argparse.ArgumentParser(
        description="RLENA: Battle Arena for RL agents"
    )
    env = ['snake', 'pommerman']
    mode = ['train', 'test', 'demo']

    common = parser.add_argument_group('common configurations')
    common.add_argument("--env", type=str, choices=env, default='pommerman')
    common.add_argument("--mode", type=str, choices=mode, default='train')
    common.add_argument("--tag", type=str, default='')
    common.add_argument("--seed", type=int, default=-1)
    common.add_argument('--render', action='store_true')
    common.add_argument('--render_interval', type=int, default=10)

    log = parser.add_argument_group("logging options")
    log.add_argument("--log_level", type=int, default=20)
    log.add_argument('--log_interval', type=int, default=10)
    log.add_argument("--save_interval", type=int, default=10000)

    dirs = parser.add_argument_group("directory configurations")
    dirs.add_argument("--log_dir", type=str, default='logs')
    dirs.add_argument("--ckpt_dir", type=str, default='logs/ckpt')
    dirs.add_argument("--load_dir", type=str)
    dirs.add_argument("--config_dir", type=str, help='configuration files : yaml, json, etc')

    algo = ['PPO', 'SACD', 'QMIX', 'COMA']
    train = parser.add_argument_group("training options")
    train.add_argument("--algo", type=str, required=True,
                       choices=algo)
    train.add_argument("--total_step", type=int, default=100)
    train.add_argument("--n_env", type=int, default=1)
    train.add_argument("--gpu_id", type=int, default=None)
    train.add_argument('--max_episode', type=int, default=20000)
    train.add_argument('--max_timestep', type=int, default=10000000)

    # env hyperparameter
    env = parser.add_argument_group("hyperparams for env")
    env.add_argument('--random_num_wall', type=bool, default=True)
    env.add_argument('--board_size', type=int, default=11)
    env.add_argument('--max_items', type=int, default=20)
    env.add_argument('--max_steps', type=int, default=800)
    env.add_argument('--max_rigid', type=int, default=1)
    env.add_argument('--max_wood', type=int, default=1)

    # Common Hyperparams for all algorithm
    hparams = parser.add_argument_group('common hyperparams for all algorithms')
    hparams.add_argument('--update_interval', type=int, default=150)
    hparams.add_argument('--eps_start', type=float, default=0.5)

    # Hyperparams for specific algorithm
    coma = parser.add_argument_group("hyperparams for coma")
    # coma.add_argument('--max_steps', type=int, default=800)
    # coma.add_argument('--update_interval', type=int, default=150)
    coma.add_argument('--eps_end', type=float, default=0.02)
    coma.add_argument('--decay_step', type=int, default=750)
    coma.add_argument('--polyak', type=float, default=0.95)
    coma.add_argument('--gamma', type=float, default=0.99)
    coma.add_argument('--lamda', type=float, default=0.99)
    coma.add_argument('--lr_ac', type=float, default=1e-5)
    coma.add_argument('--lr_cr', type=float, default=1e-6)
    coma.add_argument('--remove_stop', action='store_true')
    coma.add_argument('--onehot', action='store_true')

    qmix = parser.add_argument_group("hyperparams for qmix")
    qmix.add_argument('--pretrained', action='store_true', help="using pretrained model")

    sacd = parser.add_argument_group("hyperparams for sac discrete")
    sacd.add_argument('--cuda_device', type=int, default=0)
    sacd.add_argument('--train_interval', type=int, default=10)
    # sacd.add_argument('--update_interval', type=int, default=5)
    # sacd.add_argument('--max_step', type=int, default=1000)
    # sacd.add_argument('--dir_name', type=str, default=None)
    sacd.add_argument('--empty_map', type=bool, default=False)
    sacd.add_argument('--rand_until', type=int, default=1028)
    sacd.add_argument('--args_n_agents', type=int, default=2)

    model = parser.add_argument_group("Model options")
    model.add_argument("--model", type=str, default=None)
    model.add_argument("--checkpoint", type=str, default=None)

    args = parser.parse_args()

    if args.seed == -1:
        random.seed(None)
        args.seed = random.randrange(0, int(1e4))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.gpu_id is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # Call runner
    getattr(runners, args.algo.lower())(args)


if __name__ == '__main__':
    main()
