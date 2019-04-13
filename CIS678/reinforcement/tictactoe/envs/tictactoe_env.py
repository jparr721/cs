import gym
import numpy as np
from gym import spaces, logger
from gym.utils import seeding


class TicTacToeEnv(gym.Env):
    '''
    Description:
    Tic tac toe is a game in which the player wins over their opponent when
    they have a line of 3 in a row

    Ovservation:
    Type: Box(2)
    Num     Observation     Min     Max
    0       Num-x            0       5
    1       Num-y            0       5

    Actions:
    Type: Discrete(1)
    Num     Action
    0       Place piece

    Reward:
    Reward is +10 for winning, reward is -20 for losing

    Starting State:
    Empty board

    Episode Termination:
    No moves left
    Episode length > 200
    Winner
    '''
    def __init__(self):
        self.__version__ = '0.1.0'
        self.MAX_MOVES = 9
        self.DIAG_CHECK = [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)]
        self.BOARD_SIZE = (3, 3)
        self.GAME_BOARD = np.zeros(shape=(self.BOARD_SIZE))
        self.X_PIECE = 'x'
        self.Y_PIECE = 'y'
        self.num_x = 0
        self.num_y = 0
        self.turn = 0
        self.cur_step = -1
        self.cur_episode = 0

        # Init our gym action and env spaces
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Box(0, 5, dtype=np.int16)

        self.action_episode_memory = []

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: tuple):
        '''
        step takes an action which is the xy of the
        board location to place the piece
        '''
        assert self.action_space.contains(action),\
            '{} ({}) invalid'.format(action, type(action))
        done = False
        self.cur_step += 1
        self._mini_step(action)
        reward = self._get_reward(action)
        state = self._get_state()
        if reward == 10:
            done = True

        return state, reward, done, {}

    def reset(self):
        self.num_x = 0
        self.num_y = 0
        self.cur_step = -1
        self.game_won = False

    def render(self, mode='human'):
        print(self.GAME_BOARD)

    def _get_state(self):
        return (self.num_x, self.num_y)

    def _move_viability(self, action: tuple):
        return not self.GAME_BOARD[action[0]][action[1]]

    def _mini_step(self, action: tuple):
        if not self._move_viability:
            raise RuntimeError('Invalid move made')

        self.GAME_BOARD[action[0]][action[1]] = action[2]

    def _action(self, action: tuple):
        self.action_eposode_memory[self.cur_episode].append(action)

    def _get_reward(self, action: tuple):

        def count_piece(count_dict, piece):
            if piece in count_dict:
                count_dict[piece] += 1
            else:
                count_dict[piece] = 1

        def check_max(count_dict):
            return max(count_dict.values())

        count_dict = {}
        diag = False

        start_spot = (action[0], action[1])

        if start_spot in self.DIAG_CHECK:
            diag = True

        # If the point can win on a diag, check them
        if diag:
            # Top left corner down
            for i in range(3):
                piece = self.GAME_BOARD[i][i]
                count_piece(count_dict, piece)

            # This should ideally never go > 3...
            if check_max(count_dict) == 3:
                return 10

            # No winner? Move on...
            count_dict.clear()
            # Bottom left corner up
            j = 2
            for i in range(3):
                piece = self.GAME_BOARD[j][i]
                count_piece(count_dict, piece)
                j -= 1

            if check_max(count_dict) == 3:
                return 10

        # Check row and col of start point, we always check these
        count_dict.clear()
        for i in range(3):
            piece = self.GAME_BOARD[i][start_spot[1]]
            count_piece(count_dict, piece)

        if check_max(count_dict) == 3:
            return 10

        count_dict.clear()
        for i in range(3):
            piece = self.GAME_BOARD[start_spot[0]][i]
            count_piece(count_dict, piece)

        if check_max(count_dict) == 3:
            return 10

        return 0
