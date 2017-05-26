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

        self.screen_width = 84
        self.screen_height = 84
        self.screens = np.zeros((self.history_len, self.screen_width, self.screen_height))
        self.last_screen = np.zeros((self.screen_width, self.screen_height))

        self.ale.loadROM(rom_file)

        self.actions = self.ale.getMinimalActionSet()
        self.life_lost = False

        #from gym.envs.classic_control import rendering
        #self.viewer = rendering.SimpleImageViewer()

    def reset(self, train=True):
        if ( not train or
            not self.life_lost or
            self.ale.game_over() ):
            self.ale.reset_game()
            self.life_lost = False
        
        self.last_screen = np.zeros((self.screen_width, self.screen_height))
        for i in range(self.history_len):
            self.step(0)
            
        return self.get_screens()

    def step(self, action):
        reward = 0
        lives = self.ale.lives()
        for i in range(self.frame_skip):
          reward += self.ale.act(self.actions[action])
          self._add_screen(self._get_screen())

        state = self.get_screens()
        self.life_lost = (not (lives == self.ale.lives()))
        terminal = self.ale.game_over() or self.life_lost
        info = []

        #_, w, h = state.shape
        #ret = np.empty((w, h, 3), dtype=np.uint8)
        #ret[:, :, 0] = state[1, :, :] ; ret[:, :, 1] = state[2, :, :] ; ret[:, :, 2] = state[3, :, :]
        #self.viewer.imshow(ret) 

        return state, reward, terminal, info

    def numActions(self):
        return len(self.actions)

    def _add_screen(self, screen):
        self.screens[self.history_counter] = np.maximum(screen, self.last_screen)
        self.last_screen = screen
        self.history_counter = (self.history_counter + 1) % self.history_len

    def _get_screen(self):
        screen = self.ale.getScreenGrayscale()
        resized = cv2.resize(screen, (self.screen_width, self.screen_height))
        return resized

    def get_screens(self):
        if self.history_counter == 0:
            return self.screens
        else:
            return self.screens[self.history_counter:] + self.screens[:self.history_counter]



