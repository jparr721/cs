import gym
import numpy as np
from gym import spaces, logger
from gym.utils import seeding

LEFT = '  '


def encode(piece):
    return 1 if piece == 'x' else 2


def decode(piece):
    return 'x' if piece == 1 else 'o'


def next_piece(piece):
    return 'x' if piece == 'o' else 'o'


class TicTacToeEnv(gym.Env):
    '''
    Description:
    Tic tac toe is a game in which the player wins over their opponent when
    they have a line of 3 in a row

    Ovservation:
    Type: Discrete(10)
    Num     Observation     Min     Max
    0 - 9   The game board  0       9
    10      The piece       1       2

    Actions:
    Type: Discrete(9)
    Num     Action
    0-9     Place piece at coordinates in 3x3 plane

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
        self.starting_piece = self.X_PIECE
        self.turn = 0
        self.cur_step = -1
        self.cur_episode = 0
        self.current_piece = self.X_PIECE

        # Init our gym action and env spaces
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Discrete(10)

        self.action_episode_memory = []
        self.seed()
        self.reset()

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
        if self.done:
            return self._get_state(), self.done, None

        self.cur_step += 1
        self._mini_step(action)
        self.status = self._check_state(action)
        self.done = self.status[0]
        state = self._get_state()
        reward = 0

        if self.done:
            reward = 1 if self.status[1] == 'x' else 0

        self.current_piece = next_piece(self.current_piece)
        return state, reward, self.done, None

    def reset(self):
        self.num_x = 0
        self.num_y = 0
        self.cur_step = -1
        self.done = False
        self.GAME_BOARD = np.zeros(shape=(self.BOARD_SIZE))

    def render(self, mode='human', close=False):
        if close:
            return
        if mode == 'human':
            self._show_board()

    def _get_state(self):
        return tuple(self.GAME_BOARD), self.current_piece

    def _show_board(self):
        for i in range(0, 9, 3):
            print(LEFT + '|'.join([i for i in range(i, i + 3)]))
            if i < 6:
                print(LEFT + '-----')

    def _action(self, action: tuple):
        self.action_eposode_memory[self.cur_episode].append(action)

    def _check_state(self, action: tuple):

        def count_piece(count_dict, piece):
            if piece in count_dict:
                count_dict[piece] += 1
            else:
                count_dict[piece] = 1

        def check_max(count_dict):
            return max(count_dict.values()),
            max(count_dict, key=count_dict.get)

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
            if check_max(count_dict)[0] == 3:
                return True, check_max(count_dict)[1]

            # No winner? Move on...
            count_dict.clear()
            # Bottom left corner up
            j = 2
            for i in range(3):
                piece = self.GAME_BOARD[j][i]
                count_piece(count_dict, piece)
                j -= 1

            if check_max(count_dict) == 3:
                return True, check_max(count_dict)[1]

        # Check row and col of start point, we always check these
        count_dict.clear()
        for i in range(3):
            piece = self.GAME_BOARD[i][start_spot[1]]
            count_piece(count_dict, piece)

        if check_max(count_dict) == 3:
            return True, check_max(count_dict)[1]

        count_dict.clear()
        for i in range(3):
            piece = self.GAME_BOARD[start_spot[0]][i]
            count_piece(count_dict, piece)

        if check_max(count_dict) == 3:
            return True, check_max(count_dict)[1]

        # Game is still going...
        return False
