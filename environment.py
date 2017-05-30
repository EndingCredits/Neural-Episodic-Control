import collections
import cv2
import numpy as np

class ALEEnvironment:
    def __init__(self, rom_file, args=None):

        from ale_python_interface import ALEInterface
        self.ale = ALEInterface()
        
        # Set Env Variables
        self.ale.setInt('frame_skip', 1)
        self.ale.setFloat('repeat_action_probability', 0.0)
        self.ale.setBool('color_averaging', False)
        self.ale.setInt('random_seed', 123)
        self.ale.setBool('sound', False)
        self.ale.setBool('display_screen', False)

        self.frame_skip = 4
        self.history_len = 4
        self.history_counter = 0

        self.initial_skip_actions = 4

        self.screen_width = 84
        self.screen_height = 84
        self.screens = np.zeros((self.history_len, self.screen_width, self.screen_height))
        self.last_screen = np.zeros((self.screen_width, self.screen_height))

        self.ale.loadROM(rom_file)

        self.actions = self.ale.getMinimalActionSet()
        self.life_lost = False

    def reset(self, train=True):
        if ( not train or
            not self.life_lost or
            self.ale.game_over() ):
            self.ale.reset_game()
            self.life_lost = False
        
        self.last_screen = np.zeros((self.screen_width, self.screen_height))
        for i in range(self.initial_skip_actions):
            self.step(0)
        
        state = self.get_screens()
        print "State len " + str(len(state))
        return state

    def step(self, action):
        reward = 0
        lives = self.ale.lives()
        for i in range(self.frame_skip):
          reward += self.ale.act(self.actions[action])
        screen = self._get_screen()
        self._add_screen(screen)

        state = self.get_screens()
        self.life_lost = (not (lives == self.ale.lives()))
        terminal = self.ale.game_over() or self.life_lost
        info = []

        #print "State len " + str(len(state))
        return state, reward, terminal, info

    def numActions(self):
        return len(self.actions)

    def _add_screen(self, screen):
        self.screens[self.history_counter] = np.maximum(screen, self.last_screen)
        self.last_screen = screen
        self.history_counter = (self.history_counter + 1) % self.history_len

    def _get_screen(self):
        screen = self.ale.getScreenGrayscale()
        resized = np.array(cv2.resize(screen, (self.screen_width, self.screen_height)))
        return resized

    def get_screens(self):
        return self.screens[permutation(self.history_counter, self.history_len)]

def permutation(shift, num_elems):
    r = range(num_elems)
    if shift == 0:
      return r
    else:
      p = r[shift:] + r[:shift]
      return p




