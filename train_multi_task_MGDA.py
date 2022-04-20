import sys
# import sys
sys.path.append(".") 

import torch

import os
import time
import os.path as osp

import numpy as np
import torch.multiprocessing as mp

from torchrl.utils import get_args
from torchrl.utils import get_params
from torchrl.env import get_env

from torchrl.utils import Logger

args = get_args()
params = get_params(args.config)

import torchrl.policies as policies
import torchrl.networks as networks
from torchrl.networks.init import normal_init
from torchrl.algo import SAC
from torchrl.algo import TwinSAC
from torchrl.algo import TwinSACQ
from torchrl.algo import MTSAC
from torchrl.algo import MTSAC_MGDA
from torchrl.algo import MTMHSAC
from torchrl.collector.para import ParallelCollector
from torchrl.collector.para import AsyncParallelCollector
from torchrl.collector.para.mt import SingleTaskParallelCollectorBase
from torchrl.collector.para.async_mt import AsyncSingleTaskParallelCollector
from torchrl.collector.para.async_mt import AsyncMultiTaskParallelCollectorUniform

from torchrl.replay_buffers.shared import SharedBaseReplayBuffer
from torchrl.replay_buffers.shared import AsyncSharedReplayBuffer
import gym

from metaworld_utils.meta_env import get_meta_env

def train_multi_task(args):
    device = torch.device("cuda:{}".format(args.device) if args.cuda else "cpu")
    env, cls_dicts, cls_args = get_meta_env(params['env_name'], params['env'], params['meta_env'])
    print('args.seed', args.seed)
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.cuda:
        torch.backends.cudnn.deterministic=True
    
    buffer_param = params['replay_buffer']

    experiment_name = os.path.split(os.path.splitext(args.config)[0])[-1] if args.id is None else args.id
    logger = Logger(experiment_name, params['env_name'], args.seed, params, args.log_dir)

    params['general_setting']['env'] = env
    params['general_setting']['logger'] = logger
    params['general_setting']['device'] = device
    params['net']['base_type'] = networks.MLPBase

    mp.set_start_method('spawn', force=True)

    example_embedding = env.active_task_one_hot
    example_ob = env.reset()

    #define network structure
    if args.method == 'multitask_multihead_SAC':
        pf = policies.MultiHeadGuassianContPolicy(
            input_shape = env.observation_space.shape[0],
            output_shape = 2 * env.action_space.shape[0],
            head_num = env.num_tasks,
            **params['net']
        )

        qf1 = networks.FlattenBootstrappedNet(
            input_shape = env.observation_shape[0] + env.observation_shape[0],
            output_shape = 1,
            head_num = env.num_tasks,
            **params['net']
        )

        qf2 = networks.FlattenBootstrappedNet(
            input_shape = env.observation_shape[0] + env.observation_shape[0],
            output_shape = 1,
            head_num = env.num_tasks,
            **params['net']
        )

    elif args.method == 'multitask_SAC':
        pf = policies.GuassianContPolicy(
            input_shape = env.observation_space.shape[0],
            output_shape = 2 * env.action_space.shape[0],
            **params['net']
        )

        qf1 = networks.FlattenNet(
            input_shape = env.observation_space.shape[0] + env.action_space.shape[0],
            output_shape = 1,
            **params['net']
        )

        qf2 = networks.FlattenNet(
            input_shape = env.observation_space.shape[0] + env.action_space.shape[0],
            output_shape = 1,
            **params['net']
        )
    elif args.method == 'multitask_softmodul_SAC':
        pf = policies.ModularGuassianGatedCascadeCondContPolicy(
            input_shape = env.observation_space.shape[0],
            em_input_shape = np.prod(example_embedding.shape),
            output_shape = 2 * env.action_space.shape[0],
            **params['net']
        )

        qf1 = networks.FlattenModularGatedCascadeCondNet(
            input_shape = env.observation_space.shape[0] + env.action_space.shape[0],
            em_input_shape = np.prod(example_embedding.shape),
            output_shape = 1,
            **params['net']
        )

        qf2 = networks.FlattenModularGatedCascadeCondNet(
            input_shape = env.observation_space.shape[0] + env.action_space.shape[0],
            em_input_shape = np.prod(example_embedding.shape),
            output_shape = 1,
            **params['net']
        )

        if args.pf_snap is not None:
            pf.load_state_dict(torch.load(args.pf_snap, map_location='cpu'))
        if args.qf1_snap is not None:
            qf1.load_state_dict(torch.load(args.qf1_snap, map_location='cpu'))
        if args.qf2_snap is not None:
            qf2.load_state_dict(torch.load(args.qf2_snap, map_location='cpu'))

    else:
        raise NotImplementedError

    
    example_dict = {
        "obs": example_ob,
        "next_obs": example_ob,
        "acts": env.action_space.sample(),
        "rewards": [0],
        "terminals": [False],
        "task_idxs": [0],
        "embedding_inputs": example_embedding
    }

    replay_buffer = AsyncSharedReplayBuffer(int(buffer_param['size']), args.worker_nums)
    replay_buffer.build_by_example(example_dict)

    params['general_setting']['replay_buffer'] = replay_buffer
    epochs = params['general_setting']['pretrain_epochs'] + params['general_setting']['num_epochs']

    print(env.action_space)
    print(env.observation_space)

    params['general_setting']['collector'] = AsyncMultiTaskParallelCollectorUniform(
        env = env, pf = pf, replay_buffer = replay_buffer,
        env_cls = cls_dicts, env_args = [params['env'], cls_args, params['meta_env']],
        device = device,
        reset_idx = True,
        epoch_frames = params['general_setting']['epoch_frames'],
        max_episode_frames = params['general_setting']['max_episode_frames'],
        eval_episodes = params['general_setting']['eval_episodes'],
        worker_nums = args.worker_nums, eval_worker_nums = args.eval_worker_nums,
        train_epochs = epochs, eval_epochs = params['general_setting']['num_epochs']
    )
    params['general_setting']['batch_size'] = int(params['general_setting']['batch_size'])
    params['general_setting']['save_dir'] = osp.join(logger.work_dir, "model")

    #create agent
    agent = MTSAC_MGDA(
        pf = pf,
        qf1 = qf1,
        qf2 = qf2,
        task_nums = env.num_tasks,
        **params['sac'],
        **params['general_setting']
    )
    
    agent.train()

if __name__ == "__main__":
    train_multi_task(args)