import torch
import random
import math
from DQN import DQN
from Constant import *
from State import State
import numpy as np
from Connect4 import Connect4

class DQN_Agent:
    def __init__(self, player = 1, parameters_path = None, train = True, env= Connect4()):
        self.DQN = DQN()
        if parameters_path:
            self.DQN.load_params(parameters_path)
        self.player = player
        self.train = train
        self.env = env
        self.setTrainMode()

    def setTrainMode (self):
          if self.train:
              self.DQN.train()
          else:
              self.DQN.eval()

    def get_Action (self, state:State, epoch = 0, events= None, train = True, blue_state = None) -> int:
        actions = self.env.get_actions(self.env.state)
        if self.train and train:
            epsilon = self.epsilon_greedy(epoch)
            rnd = random.random()
            if rnd < epsilon:
                return random.choice(actions)

        if self.player == 1:
            state_tensor = state.toTensor()
        elif not blue_state:
            blue_state = state.reverse()
            state_tensor = blue_state.toTensor()
        else:
            state_tensor = blue_state.toTensor()

        action_np = np.array(actions)
        action_tensor = torch.from_numpy(action_np).unsqueeze(1).to(dtype=torch.float32)
        expand_state_tensor = state_tensor.unsqueeze(0).repeat((len(action_tensor),1))

        with torch.no_grad():
            Q_values = self.DQN(expand_state_tensor, action_tensor)
        max_index = torch.argmax(Q_values)
        return actions[max_index]

    def get_Actions (self, states_tensor: State, dones) -> torch.tensor:
        actions = []
        for i in range(len(states_tensor)):
            if dones[i].item():
                actions.append(0)
            else:
                actions.append(self.get_Action(State.tensorToState(state_tensor=states_tensor[i],player=self.player), train=False))
        return torch.tensor(actions, dtype=torch.float32)

    def epsilon_greedy(self,epoch, start = epsilon_start, final=epsilon_final, decay=epsilon_decay):
        res = final + (start - final) * math.exp(-1 * epoch/decay)
        return res
    
    def loadModel (self, file):
        self.model = torch.load(file)
    
    def save_param (self, path):
        self.DQN.save_params(path)

    def load_params (self, path):
        self.DQN.load_params(path)

    def __call__(self, events= None, state=None):
        return self.get_Action(state)