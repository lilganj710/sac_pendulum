import sys
from copy import deepcopy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import gymnasium as gym
import random
import matplotlib
import matplotlib.pyplot as plt
import pdb
import time
from torch.distributions import Normal

class ReplayBuffer:
    def __init__(self,memory_capacity=1000000,batch_size=64,num_actions=1,num_states=3):
        self.memory_capacity=memory_capacity
        self.num_states=num_states
        self.num_actions=num_actions
        self.batch_size=batch_size
        self.buffer_counter=0
        self.state_buffer=np.zeros((self.memory_capacity,self.num_states))
        self.action_buffer=np.zeros((self.memory_capacity,self.num_actions))
        self.reward_buffer=np.zeros(self.memory_capacity)
        self.next_state_buffer=np.zeros((self.memory_capacity,self.num_states))
        self.done_buffer=np.zeros(self.memory_capacity)

    def record(self,observation,action,reward,next_observation,done):
        index = self.buffer_counter % self.memory_capacity
        self.state_buffer[index] = observation
        self.action_buffer[index] = action
        self.reward_buffer[index] = reward
        self.next_state_buffer[index] = next_observation
        self.done_buffer[index] = done
        self.buffer_counter += 1

    def sample(self):
        range1 = min(self.buffer_counter, self.memory_capacity)
        indices = np.random.randint(0, range1, size=self.batch_size)
        states = torch.tensor(self.state_buffer[indices], dtype=torch.float32)
        actions = torch.tensor(self.action_buffer[indices], dtype=torch.float32)
        rewards = torch.tensor(self.reward_buffer[indices], dtype=torch.float32)
        next_states = torch.tensor(self.next_state_buffer[indices], dtype=torch.float32)
        dones = torch.tensor(self.done_buffer[indices], dtype=torch.float32)
        return states,actions,rewards,next_states,dones
    
def create_network(input_dims: int, output_dims: int, hidden_size: tuple[int, ...]) -> nn.Sequential:
    '''Creates Sequential object representing a feedforward net. ReLU activations (but no ReLU after the last layer)
    :param input_dims: number of input dimensions
    :param output_dims: number of output dimensions
    :param hidden_size: tuple of number of neurons in each hidden layer'''
    first_layer = nn.Linear(input_dims, hidden_size[0])
    last_layer = nn.Linear(hidden_size[-1], output_dims)
    hidden_layers = [[nn.Linear(hidden_size[i], hidden_size[i+1]), nn.ReLU()] for i in range(len(hidden_size)-1)]
    flattened_hidden_layers = [layer for linear_relu in hidden_layers for layer in linear_relu]
    sequential_object = nn.Sequential(
        first_layer,
        nn.ReLU(),
        *flattened_hidden_layers,
        last_layer
    )
    return sequential_object

class Critic(nn.Module):
    def __init__(self,num_states,num_actions,action_bound,learning_rate,
                 hidden_size: tuple[int, ...]=(256,256,)):
        super(Critic,self).__init__()
        self.num_actions=num_actions
        self.num_states=num_states
        self.network = create_network(num_states+num_actions, 1, hidden_size)

    def forward(self, s, a):
        combined = torch.cat([s, a], dim=-1)
        return self.network(combined)

class Actor(nn.Module):
    def __init__(self,num_states,num_actions,learning_rate,action_bound,
                 hidden_size: tuple[int, ...]=(256,256,)):
        super(Actor,self).__init__()
        self.num_states=num_states
        self.num_actions=num_actions
        self.action_bound=action_bound

        self.gaussian_params_network = create_network(num_states, num_actions*2, hidden_size)
        self.min_log_std=-20
        self.max_log_std=2

    def forward(self,state):
        gaussian_params = self.gaussian_params_network(state)
        mu = gaussian_params[..., :self.num_actions]
        log_std = gaussian_params[..., self.num_actions:]
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        return mu, log_std

class Agent:
    def __init__(self, env: gym.Env):
        self.env=env
        self.state_dimension=self.env.observation_space.shape[0]
        self.action_dimension=self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high[0]
        self.buffer=ReplayBuffer()
        self.learning_rate1 = 3e-4
        self.learning_rate2 = 3e-4
        self.tau=.005
        self.gamma=.99
        self.alpha=0.2

        self.actor=Actor(self.state_dimension,self.action_dimension,self.learning_rate1,self.action_bound)
        self.critic=Critic(self.state_dimension,self.action_dimension,self.action_bound,self.learning_rate2)
        self.target_critic = deepcopy(self.critic)
        #self.target_critic=Critic(self.state_dimension,self.action_dimension,self.action_bound,self.learning_rate2)
        #self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_optimizer=optim.Adam(self.actor.parameters(),lr=self.learning_rate1)
        self.critic_optimizer=optim.Adam(self.critic.parameters(),lr=self.learning_rate2)

        self.critic2=Critic(self.state_dimension,self.action_dimension,self.action_bound,self.learning_rate2)
        self.target_critic2 = deepcopy(self.critic2)
        #self.target_critic2=Critic(self.state_dimension,self.action_dimension,self.action_bound,self.learning_rate2)
        #self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.critic2_optimizer=optim.Adam(self.critic2.parameters(),lr=self.learning_rate2)

        self.sp = nn.Softplus()

    def action_w_probs(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        '''Given a batch of states, sample actions. Also, return log probs of those actions'''
        mu, log_std = self.actor(state)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mu,std)

        unclipped_actions = normal.rsample()
        action = self.action_bound * torch.tanh(unclipped_actions)

        log_probs = normal.log_prob(unclipped_actions).sum(axis=-1,keepdim=True)
        #transformation_term = torch.log(1-action.pow(2)+1e-6)
        transformation_term = 2 * (np.log(2) - unclipped_actions - self.sp(-2*unclipped_actions)).sum(axis=-1, keepdim=True) + np.log(self.action_bound)*self.action_dimension
        log_probs -= transformation_term

        #print(mu[0], std[0], unclipped_actions[0], action[0], log_probs[0])

        return action, log_probs
    
    def soft_update(self):
        for target_param,param in zip(self.target_critic.parameters(),self.critic.parameters()):
            target_param.data.copy_(self.tau*param.data+(1-self.tau)*target_param.data)
        for target_param, param in zip(self.target_critic2.parameters(),self.critic2.parameters()):
            target_param.data.copy_(self.tau*param.data+(1-self.tau)*target_param.data)
    
    def train(self,max_step,max_episode):
        theta_values=[]
        time_values=[]
        for episode in range(max_episode):
            state,_=self.env.reset()
            print("/////////////////////")
            print("episode",episode)
            total_reward = 0
            q_losses = []
            actor_losses = []
            for step in range(max_step):
                action, _ = self.action_w_probs(torch.FloatTensor(state))
                action = action.detach().numpy()
                #action=np.clip(action,-self.action_bound,self.action_bound)
                if False:
                    if step == 0:
                        print(f'action: ', end='')
                    print(action, end=', ')
                #print('action',action.shape)
                next_state,reward,done,trunc,info=self.env.step(action)
                total_reward += reward
                self.buffer.record(state,action,reward,next_state,done)

                if episode == 0:
                    #Populate the buffer a bit before sampling from it
                    continue

                states,actions,rewards,next_states,dones=self.buffer.sample()
                rewards = torch.unsqueeze(rewards, 1)
                dones = torch.unsqueeze(dones, 1)
                
                q1=self.critic(states,actions)
                q2=self.critic2(states,actions)
                #print('q1',q1)
                with torch.no_grad():
                    next_action, next_log_probs = self.action_w_probs(next_states)
                    #print('next_action',next_action)
                    q1_next_target=self.target_critic(next_states,next_action)
                    q2_next_target=self.target_critic2(next_states,next_action)
                    q_next_target=torch.min(q1_next_target,q2_next_target)
                    #print('q1_next',q1_next_target)
                    value_target = rewards + (1-dones) * self.gamma * (q_next_target - self.alpha*next_log_probs)
                    #print(f'{rewards.shape=}, {q_next_target.shape=}, {next_log_probs.shape=}, {value_target.shape=}')
                    #print('value_target',value_target)
                l1_q_loss = False
                if l1_q_loss:
                    q1_loss=(torch.sqrt(1 + (q1-value_target)**2) - 1).mean()
                    q2_loss=(torch.sqrt(1 + (q2-value_target)**2) - 1).mean()
                else:
                    q1_loss=((q1-value_target)**2).mean()
                    q2_loss=((q2-value_target)**2).mean()
                loss_q=q2_loss+q1_loss 
                self.critic_optimizer.zero_grad()
                self.critic2_optimizer.zero_grad()
                loss_q.backward()
                self.critic_optimizer.step()
                self.critic2_optimizer.step()
                q_losses.append(loss_q.item())
                
                self.actor_optimizer.zero_grad()
                actions_pred, log_pred = self.action_w_probs(states)
                #print(f'{states.shape=}, {actions_pred.shape=}, {log_pred.shape=}')
                q1_pred=self.critic(states,actions_pred)
                q2_pred=self.critic2(states,actions_pred)
                q_pred=torch.min(q1_pred,q2_pred)
                actor_loss=(self.alpha*log_pred-q_pred).mean()
                actor_loss.backward()
                self.actor_optimizer.step()
                actor_losses.append(actor_loss.item())

                self.soft_update()
                if done:
                    break
                state=next_state
            print(f'\n{total_reward=}, {np.mean(q_losses)=}, {np.mean(actor_losses)=}\n')

    def save_model(self,actor_path,critic_path1,critic_path2):
        torch.save(self.actor.state_dict(),actor_path,)
        torch.save(self.critic.state_dict(),critic_path1,)
        torch.save(self.critic2.state_dict(),critic_path2,)
        print("model saved")
    def load_model(self,actor_path,critic_path1,critic_path2):
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path1))
        self.critic2.load_state_dict(torch.load(critic_path2))
        print("model loaded")
            
    
def main():
    #env=gym.make('Pendulum-v1',render_mode='human')
    env=gym.make('Pendulum-v1')
    max_episode=200
    max_step=200
    agent=Agent(env)

    actor_path=f'{os.path.dirname(__file__)}/saved_models/actor_modelSAC.pth'
    critic_path1=f'{os.path.dirname(__file__)}/saved_models/critic_modelSAC1.pth'
    critic_path2=f'{os.path.dirname(__file__)}/saved_models/critic_modelSAC2.pth'
    if not os.path.exists(os.path.dirname(actor_path)):
        os.makedirs(os.path.dirname(actor_path))
    #agent.load_model(actor_path,critic_path1,critic_path2)

    agent.train(max_step,max_episode)
    agent.save_model(actor_path,critic_path1,critic_path2)
    if False:
        max_step=300
        #reward=agent.test(max_step)
        print(f'Reward from test:{reward}')

    env.close()

if __name__ == '__main__':
    main()