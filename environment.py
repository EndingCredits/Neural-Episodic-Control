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
        self.initial_skip_actions = 5
        self.screen_width = 84
        self.screen_height = 84

        self.last_screen = np.zeros((self.screen_width, self.screen_height))

        self.ale.loadROM(rom_file)

        self.actions = self.ale.getMinimalActionSet()
        self.life_lost = False

        self.training = True 

    def reset(self, train=True):
        self.training = train
        if ( self.ale.game_over()
            or not (train and self.life_lost) ):
            self.ale.reset_game()
            self.last_screen.fill(0.0)
        
            for i in range(self.initial_skip_actions):
               self.step(0)
        
        state = self._get_screen()#self.get_screens()
        return state

    def step(self, action):
        reward = 0
        lives = self.ale.lives()
        for i in range(self.frame_skip):
          reward += self.ale.act(self.actions[action])
        screen = self._get_screen()
        #self._add_screen(screen)

        state = screen #self.get_screens()
        self.life_lost = (not (lives == self.ale.lives()))
        terminal = self.ale.game_over() or (self.life_lost and self.training)
        info = []

        return state, reward, terminal, info

    def numActions(self):
        return len(self.actions)

    def _get_screen(self):
        screen = self.ale.getScreenGrayscale()
        resized = np.array(cv2.resize(screen, (self.screen_width, self.screen_height)))
        porcessed_screen = np.maximum(resized, self.last_screen)
        self.last_screen = resized
        return resized







