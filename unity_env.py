import collections
import numpy as np
from unity_client.fetcher import Fetcher
# This can be found here: https://github.com/chetan51/unity-api
import time

class UnityEnvironment:
    def __init__(self, ):
        self.fetcher = Fetcher()
        
        self.actions = [ [ 0, 0],
                         [ 1, 0],
                         [ 1, 1],
                         [ 0, 1],
                         [-1, 1],
                         [-1, 0],
                         [-1,-1],
                         [ 0,-1],
                         [ 1,-1] ]
        self.curr_score = 0
        self.curr_step = 0


    def reset(self):
        state, _, _ = self._get_obs()
        self.curr_score = 0
        self.curr_step = 0
        
        self.fetcher.inputData = { 'Reset': True }
        self.fetcher.sync()
        time.sleep(0.04)
        state, _, _, _ = self.step(0)
        
        return state


    def step(self, action):
        hor = self.actions[action][0]
        vert = self.actions[action][1]
        self.fetcher.inputData = { 'Horizontal': hor, 'Vertical': vert }
        self.fetcher.sync()
        
        time.sleep(0.04)
        self.curr_step +=1
        
        state, reward, terminal = self._get_obs()
        
        if self.curr_step > 1000:
            terminal = True

        return state, reward, terminal, []


    def numActions(self):
        return len(self.actions)

    def _get_obs(self):
    
        data=None ; count = 0
        while data is None:
          count += 1
          data = self.fetcher.sync()
          time.sleep(0.001)
          if count > 1000:
              print("Error getting data from server.")
              
        new_score = data.get('Score')
        if new_score is not None:
            reward = max(new_score - self.curr_score, 0)
            self.curr_score = new_score
        else:
            reward = 0
        
        terminal = data.get('Terminal') or False
        
        objects = []
        for k, v in data.items():
          if 'obj' in k:
            class_one_hot = [ 1.0, 0.0] if (v['tag'] == "Player") else [0.0, 1.0]
            pos = v['position'] ; vel = v['velocity']
            pos = [ pos['y'], pos['x'], pos['z'] ]
            vel = [ vel['y'], vel['x'], vel['z'] ]
            objects.append(np.array( class_one_hot + pos + vel ))
            
        if (len(objects) == 0):
            objects = np.zeros((1,8))
        
        return objects, reward, terminal 

