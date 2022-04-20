from .twin_sac_q import TwinSACQ
import sys
import click
import json
import datetime

import copy
import torch
import numpy as np

import torchrl.policies as policies
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils import data
#import torchvision
import types
from tqdm import tqdm
from tensorboardX import SummaryWriter

from multiobj_optimization.min_norm_solvers import MinNormSolver, gradient_normalizers
from multiobj_optimization.min_norm_solvers_numpy import MinNormSolverNumpy

class MTSAC_MGDA(TwinSACQ):
    """"
    Support Different Temperature for different tasks
    """
    def __init__(self, task_nums,
                 temp_reweight=False,
                 grad_clip=True,
                 **kwargs):
        super().__init__(**kwargs)

        self.task_nums = task_nums
        if self.automatic_entropy_tuning:
            self.log_alpha = torch.zeros(self.task_nums).to(self.device)
            self.log_alpha.requires_grad_()
            self.alpha_optimizer = self.optimizer_class(
                [self.log_alpha],
                lr=self.plr,
            )
        self.sample_key = ["obs", "next_obs", "acts", "rewards",
                           "terminals",  "task_idxs"]

        self.pf_flag = isinstance(self.pf,
                                  policies.EmbeddingGuassianContPolicyBase)

        self.idx_flag = isinstance(self.pf, policies.MultiHeadGuassianContPolicy)

        assert self.pf_flag == False
        assert self.idx_flag == False

        self.temp_reweight = temp_reweight
        if self.pf_flag:
            self.sample_key.append("embedding_inputs")
        self.grad_clip = grad_clip

    def update(self, batch):
        self.training_update_num += 1
        obs = batch['obs']
        actions = batch['acts']
        next_obs = batch['next_obs']
        rewards = batch['rewards']
        terminals = batch['terminals']
        task_idx = batch['task_idxs']

        if self.pf_flag:
            embedding_inputs = batch["embedding_inputs"]

        if self.idx_flag:
            task_idx = batch['task_idxs']
        
        
        #print(task_idx)
        rewards = torch.Tensor(rewards).to(self.device)
        terminals = torch.Tensor(terminals).to(self.device)
        obs = torch.Tensor(obs).to(self.device)
        actions = torch.Tensor(actions).to(self.device)
        next_obs = torch.Tensor(next_obs).to(self.device)
        task_idx = torch.Tensor(task_idx).to(self.device).int()

        if self.pf_flag:
            embedding_inputs = torch.Tensor(embedding_inputs).to(self.device)

        if self.idx_flag:
            task_idx    = torch.Tensor(task_idx).to( self.device ).long()

        assert len(obs) == len(actions) == len(next_obs) == len(rewards) == len(terminals) == len(task_idx) 

        obs_list = [torch.tensor([]).to(self.device) for i in range(self.task_nums)]
        actions_list = [torch.tensor([]).to(self.device) for i in range(self.task_nums)]
        next_obs_list = [torch.tensor([]).to(self.device) for i in range(self.task_nums)]
        rewards_list = [torch.tensor([]).to(self.device) for i in range(self.task_nums)]
        terminals_list = [torch.tensor([]).to(self.device) for i in range(self.task_nums)]
        task_idx_list = [torch.tensor([]).to(self.device) for i in range(self.task_nums)]
        if self.pf_flag:
            embedding_inputs_list = [torch.tensor([]).to(self.device) for i in range(self.task_nums)]

        #print('task_idx.shape', task_idx.shape)
        #print('rewards.shape', rewards.shape)
        #print('terminals.shape', terminals.shape)
        #print('obs.shape', obs.shape)
        #print('actions.shape', actions.shape)
        #print('next_obs.shape', next_obs.shape)


        for j in range(task_idx.shape[0]):
            for i in range(task_idx.shape[1]):
                obs_list[task_idx[j,i,0]] = torch.cat((obs_list[task_idx[j,i,0]], obs[j,i].unsqueeze(0)), dim=0)
                actions_list[task_idx[j,i,0]] = torch.cat((actions_list[task_idx[j,i,0]], actions[j,i].unsqueeze(0)), dim=0)
                next_obs_list[task_idx[j,i,0]] = torch.cat((next_obs_list[task_idx[j,i,0]], next_obs[j,i].unsqueeze(0)), dim=0)
                rewards_list[task_idx[j,i,0]] = torch.cat((rewards_list[task_idx[j,i,0]], rewards[j,i].unsqueeze(0)), dim=0)
                terminals_list[task_idx[j,i,0]] = torch.cat((terminals_list[task_idx[j,i,0]], terminals[j,i].unsqueeze(0)), dim=0)
                task_idx_list[task_idx[j,i,0]] = torch.cat((task_idx_list[task_idx[j,i,0]], task_idx[j,i].unsqueeze(0)), dim=0)
                if self.pf_flag:
                    embedding_inputs_list[task_idx[j,i,0]] = torch.cat((embedding_inputs_list[task_idx[j,i,0]], embedding_inputs[j,i].unsqueeze(0)), dim=0)

        self.pf.train()
        self.qf1.train()
        self.qf2.train()
        #print('obs_list.len', len(obs_list))

        #loss_data_alpha = {}
        #loss_alpha = {}
        #grads_alpha = {}
        #scale_alpha = {}

        loss_data_pf = {}
        loss_pf = {}
        grads_pf = {}
        scale_pf = {}

        loss_data_qf1 = {}
        loss_qf1 = {}
        grads_qf1 = {}
        scale_qf1 = {}

        loss_data_qf2 = {}
        loss_qf2 = {}
        grads_qf2 = {}
        scale_qf2 = {}    

        log_probs_display = []

        for task_id in range(self.task_nums):
            """
            Policy operations.
            """
            if self.idx_flag:
                sample_info = self.pf.explore(obs_list[task_id], task_idx_list[task_id],
                                            return_log_probs=True)
            else:
                if self.pf_flag:
                    sample_info = self.pf.explore(obs_list[task_id], embedding_inputs_list[task_id],
                                                return_log_probs=True)
                else:
                    sample_info = self.pf.explore(obs_list[task_id], return_log_probs=True)

            mean = sample_info["mean"]
            log_std = sample_info["log_std"]
            new_actions = sample_info["action"]
            log_probs = sample_info["log_prob"]
            log_probs_display.append(log_probs.detach())


            if self.idx_flag:
                q1_pred = self.qf1([obs_list[task_id], actions_list[task_id]], task_idx_list[task_id])
                q2_pred = self.qf2([obs_list[task_id], actions_list[task_id]], task_idx_list[task_id])
            else:
                if self.pf_flag:
                    q1_pred = self.qf1([obs_list[task_id], actions_list[task_id]], embedding_inputs_list[task_id])
                    q2_pred = self.qf2([obs_list[task_id], actions_list[task_id]], embedding_inputs_list[task_id])
                else:
                    q1_pred = self.qf1([obs_list[task_id], actions_list[task_id]])
                    q2_pred = self.qf2([obs_list[task_id], actions_list[task_id]])

            reweight_coeff = 1
            if self.automatic_entropy_tuning:
                """
                Alpha Loss
                """
                #batch_size_task = task_idx.shape[0]
                #log_alphas = (self.log_alpha.unsqueeze(0)).expand(
                #                (batch_size_task, 1))
                #log_alphas = self.log_alpha.unsqueeze(0)
                #log_alphas = log_alphas.unsqueeze(-1)
                # log_alphas = log_alphas.gather(1, task_idx)

                #alpha_loss = -(log_alphas *
                #            (log_probs + self.target_entropy).detach()).mean()

                #self.alpha_optimizer.zero_grad()
                #alpha_loss.backward()
                #self.alpha_optimizer.step()
                #loss_data_alpha[task_id] = alpha_loss.data
                #loss_alpha[task_id] = alpha_loss
                #alpha_loss.backward()
                #grads_alpha[task_id] = []
                #if self.log_alpha.grad is not None:
                #    grads_alpha[task_id].append(Variable(self.log_alpha.grad.data.clone(), requires_grad=False))

                #alphas = (self.log_alpha.exp().detach()).unsqueeze(0)
                #print('alphas.shape', alphas.shape)
                #alphas = alphas.expand((batch_size_task, 1)).unsqueeze(-1)
                #alphas = alphas.unsqueeze(-1)
                # (batch_size, 1)
                #if self.temp_reweight:
                #    softmax_temp = F.softmax(-self.log_alpha.detach()).unsqueeze(0)
                #    reweight_coeff = softmax_temp.expand((batch_size_task,
                #                                        1))
                #    reweight_coeff = reweight_coeff.unsqueeze(-1) * self.task_nums
                ##TODO: Frank-Wolfe iteration to compute scales
            else:
                alphas = 1
                alpha_loss = 0

            with torch.no_grad():
                if self.idx_flag:
                    target_sample_info = self.pf.explore(next_obs_list[task_id],
                                                        task_idx_list[task_id],
                                                        return_log_probs=True)
                else:
                    if self.pf_flag:
                        target_sample_info = self.pf.explore(next_obs_list[task_id],
                                                            embedding_inputs_list[task_id],
                                                            return_log_probs=True)
                    else:
                        target_sample_info = self.pf.explore(next_obs_list[task_id],
                                                            return_log_probs=True)

                target_actions = target_sample_info["action"]
                target_log_probs = target_sample_info["log_prob"]
        
                #print('target_actions.shape', target_actions.shape)
                #print('target_log_probs.shape', target_log_probs.shape)

                if self.idx_flag:
                    target_q1_pred = self.target_qf1([next_obs_list[task_id], target_actions],
                                                    task_idx_list[task_id])
                    target_q2_pred = self.target_qf2([next_obs_list[task_id], target_actions],
                                                    task_idx_list[task_id])
                else:
                    if self.pf_flag:
                        target_q1_pred = self.target_qf1([next_obs_list[task_id], target_actions],
                                                        embedding_inputs_list[task_id])
                        target_q2_pred = self.target_qf2([next_obs_list[task_id], target_actions],
                                                        embedding_inputs_list[task_id])
                    else:
                        target_q1_pred = self.target_qf1([next_obs_list[task_id], target_actions])
                        target_q2_pred = self.target_qf2([next_obs_list[task_id], target_actions])

                min_target_q = torch.min(target_q1_pred, target_q2_pred)
                target_v_values = min_target_q - alphas * target_log_probs
                #print('target_v_values.shape', target_v_values.shape)
                """
                QF Loss
                """
                # q_target = rewards + (1. - terminals) * self.discount * target_v_values
                # There is no actual terminate in meta-world -> just filter all time_limit terminal

                q_target = rewards_list[task_id] + self.discount * target_v_values

            ##TODO: Scaled back-propagation
            qf1_loss = (reweight_coeff *
                        ((q1_pred - q_target.detach()) ** 2)).mean()
            qf2_loss = (reweight_coeff *
                        ((q2_pred - q_target.detach()) ** 2)).mean()

            loss_data_qf1[task_id] = qf1_loss.data
            loss_qf1[task_id] = qf1_loss
            #self.qf1_optimizer.zero_grad()
            qf1_loss.backward(retain_graph=True)
            grads_qf1[task_id] = []
            for param in self.qf1.parameters():
                if param.grad is not None:
                    grads_qf1[task_id].append(Variable(param.grad.data.clone(), requires_grad=False))

            loss_data_qf2[task_id] = qf2_loss.data
            loss_qf2[task_id] = qf2_loss
            #self.qf2_optimizer.zero_grad()
            qf2_loss.backward(retain_graph=True)
            grads_qf2[task_id] = []
            for param in self.qf2.parameters():
                if param.grad is not None:
                    grads_qf2[task_id].append(Variable(param.grad.data.clone(), requires_grad=False))
            
            assert q1_pred.shape == q_target.shape, print(q1_pred.shape, q_target.shape)
            assert q2_pred.shape == q_target.shape, print(q1_pred.shape, q_target.shape)

            if self.idx_flag:
                q_new_actions = torch.min(
                    self.qf1([obs_list[task_id], new_actions], task_idx_list[task_id]),
                    self.qf2([obs_list[task_id], new_actions], task_idx_list[task_id]))
            else:
                if self.pf_flag:
                    q_new_actions = torch.min(
                        self.qf1([obs_list[task_id], new_actions], embedding_inputs_list[task_id]),
                        self.qf2([obs_list[task_id], new_actions], embedding_inputs_list[task_id]))
                else:
                    q_new_actions = torch.min(
                        self.qf1([obs_list[task_id], new_actions]),
                        self.qf2([obs_list[task_id], new_actions]))
            """
            Policy Loss
            """
            if not self.reparameterization:
                raise NotImplementedError
            else:
                assert log_probs.shape == q_new_actions.shape
                policy_loss = (reweight_coeff *
                            (alphas * log_probs - q_new_actions)).mean()

            std_reg_loss = self.policy_std_reg_weight * (log_std**2).mean()
            mean_reg_loss = self.policy_mean_reg_weight * (mean**2).mean()

            policy_loss += std_reg_loss + mean_reg_loss

            loss_data_pf[task_id] = policy_loss.data
            loss_pf[task_id] = policy_loss
            #self.pf_optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)
            grads_pf[task_id] = []
            for param in self.pf.parameters():
                if param.grad is not None:
                    grads_pf[task_id].append(Variable(param.grad.data.clone(), requires_grad=False))


        """
        Update Networks
        """
        # Frank-Wolfe iteration to compute scales.
        #sol_alpha, min_norm = MinNormSolver.find_min_norm_element([grads_alpha[t] for t in range(self.task_nums)])
        sol_pf, min_norm = MinNormSolver.find_min_norm_element_FW([grads_pf[t] for t in range(self.task_nums)])
        sol_qf1, min_norm = MinNormSolver.find_min_norm_element_FW([grads_qf1[t] for t in range(self.task_nums)])
        sol_qf2, min_norm = MinNormSolver.find_min_norm_element_FW([grads_qf2[t] for t in range(self.task_nums)])

        #('sol_pf', sol_pf)
        #print('sol_qf1', sol_qf1)
        #print('sol_qf2', sol_qf2)

        for t in range(self.task_nums):
            #scale_alpha[t] = float(sol_alpha[t])     
            scale_pf[t] = float(sol_pf[t])
            scale_qf1[t] = float(sol_qf1[t])
            scale_qf2[t] = float(sol_qf2[t]) 

        for t in range(self.task_nums):
            if t > 0:
                #alpha_loss = alpha_loss + scale_alpha[t] * loss_alpha[t]
                policy_loss = policy_loss + scale_pf[t] * loss_pf[t]
                qf1_loss = qf1_loss + scale_qf1[t] * loss_qf1[t]
                qf2_loss = qf2_loss + scale_qf2[t] * loss_qf2[t]
            else:
                #alpha_loss = scale_alpha[t] * loss_alpha[t]
                policy_loss = scale_pf[t] * loss_pf[t]
                qf1_loss = scale_qf1[t] * loss_qf1[t]
                qf2_loss = scale_qf2[t] * loss_qf2[t]                

        #self.alpha_optimizer.zero_grad()
        #alpha_loss.backward()
        #self.alpha_optimizer.step()
   
        self.pf_optimizer.zero_grad()
        policy_loss.backward()
        if self.grad_clip:
            pf_norm = torch.nn.utils.clip_grad_norm_(self.pf.parameters(), 1)
        self.pf_optimizer.step()

        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        if self.grad_clip:
            qf1_norm = torch.nn.utils.clip_grad_norm_(self.qf1.parameters(), 1)
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        if self.grad_clip:
            qf2_norm = torch.nn.utils.clip_grad_norm_(self.qf2.parameters(), 1)
        self.qf2_optimizer.step()

        self._update_target_networks()

        # Information For Logger
        info = {}
        info['Reward_Mean'] = rewards.mean().item()

        if self.automatic_entropy_tuning:
            for i in range(self.task_nums):
                info["alpha_{}".format(i)] = self.log_alpha[i].exp().item()
            info["Alpha_loss"] = alpha_loss.item()
        info['Training/policy_loss'] = policy_loss.item()
        info['Training/qf1_loss'] = qf1_loss.item()
        info['Training/qf2_loss'] = qf2_loss.item()

        if self.grad_clip:
            info['Training/pf_norm'] = pf_norm.item()
            info['Training/qf1_norm'] = qf1_norm.item()
            info['Training/qf2_norm'] = qf2_norm.item()
       
        # The following record is only the record of last task 
        #TODO: record the data of all tasks
        info['log_std/mean'] = log_std.mean().item()
        info['log_std/std'] = log_std.std().item()
        info['log_std/max'] = log_std.max().item()
        info['log_std/min'] = log_std.min().item()

        #log_probs_display = log_probs.detach()
        #log_probs_display = (log_probs_display.mean(0)).squeeze(1)
        """
        for i in range(self.task_nums):
            info["log_prob_{}".format(i)] = log_probs_display[i].item()
        """
        info['log_probs/mean'] = log_probs.mean().item()
        info['log_probs/std'] = log_probs.std().item()
        info['log_probs/max'] = log_probs.max().item()
        info['log_probs/min'] = log_probs.min().item()
        

        info['mean/mean'] = mean.mean().item()
        info['mean/std'] = mean.std().item()
        info['mean/max'] = mean.max().item()
        info['mean/min'] = mean.min().item()

        return info

    def update_per_epoch(self):
        for _ in range(self.opt_times):
            batch = self.replay_buffer.random_batch(self.batch_size,
                                                    self.sample_key,
                                                    reshape=False)
            infos = self.update(batch)
            self.logger.add_update_info(infos)
