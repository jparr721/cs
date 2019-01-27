import hlt
from hlt import constants
from hlt.positionals import Direction
from hlt.positionals import Position
import random
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.autograd import Variable
import matplotlib.pyplot as plt
import random
import numpy as np
from itertools import count
from policy_net import PolicyNet
import sys

"""

This script loads in the model and collects training data.

"""
episode = sys.argv[1]
no_update = sys.argv[2].startswith("n")
data_path = 'data/batch_data'
ship_vision_range = 16
halite_threshold = 500
rscale = 10

# Utilities
def cell_data(cell):
    return [cell.halite_amount]

def get_vision(ship, sight_range):    
    game_map = game.game_map
    sight = []
    for x in range(ship.position.x-sight_range,ship.position.x+sight_range):
        for y in range(ship.position.y-sight_range,ship.position.y+sight_range):
            sight += cell_data(game_map[game_map.normalize(Position(x, y))])
    return sight

def nav_dir(ship, destination):
    return game.game_map.naive_navigate(ship, destination)

def select_direction(state):
    state = torch.FloatTensor(state).cuda()
    state = Variable(state)
    probs = policy_net(state)
    m = Categorical(probs)
    action_sample = m.sample()
    direction = [
            Direction.North,
            Direction.South,
            Direction.East,
            Direction.West,
            Direction.Still
        ][int(action_sample)]
    return direction, action_sample

def norm(game_map, position):
    return game_map.normalize(position)

def is_safe(position):
    position = norm(game_map, position)
    if position in unsafe_positions:
        return False
    return True

policy_net = PolicyNet()
policy_net.cuda()

# Load existing / save new model
try:
    policy_net.load_state()
    logging.info("loaded state")
except:
    policy_net.save_state()
    
# Set up variables
ship_data = {}
last_halite = 0
last_action_sample = None
last_state = None
unsafe_positions = []
t = 0

# Game loop
game = hlt.Game()
game.ready("MyPythonBot")

while True:
    t += 1
    game.update_frame()
    me = game.me
    game_map = game.game_map
    shipyard = me.shipyard.position
    commands = []

    unsafe_positions = [norm(game_map, ship.position) for ship in me.get_ships()]

    for ship in me.get_ships():
        if ship.id in ship_data:
            continue
        # last state, last action sample, last position, go home, last halite
        ship_data[ship.id] = [None, None, None, False, 0]

    for sid, values in ship_data.items():
        if not values:
            continue
        
        last_state, last_action_sample, last_position, go_home, last_halite = values
        
        # Calculate reward
        last_reward = 0
        if not me.has_ship(sid): # Dead robot
            last_reward = 0 # Punishing robots to death lead to them giving up
            logging.info("\n\nDEAD SHIP\n\n")
        else:
            last_reward = 1*rscale if last_halite >= halite_threshold else 0
            
        if not last_state or go_home:
            continue
            
        # Write to file
        if not no_update:
            with open(data_path, "a") as f:
                f.write("{}|{}|{}|{}|{}|{}\n".format(
                    episode,
                    t,
                    sid,
                    last_reward,
                    last_action_sample,
                    last_state
                ))

        if not me.has_ship(sid):
            ship_data[sid] = None
            
    for ship in me.get_ships():
        state = None
        direction = None
        action_sample = None
        
        if ship.halite_amount > halite_threshold:
            ship_data[ship.id][3] = True

        if ship.position == shipyard:
            ship_data[ship.id][3] = False
        
        if ship_data[ship.id][3]:
            direction = nav_dir(ship, shipyard)
        else:
            # Forward pass NN for this turns action, collect data
            state = get_vision(ship, ship_vision_range)
            direction, action_sample = select_direction(state)

        # Prevent collisions
        next_pos = ship.position.directional_offset(direction)
        if not is_safe(next_pos):
            direction = Direction.Still
        elif direction != Direction.Still:
            pass
            #unsafe_positions.remove(ship.position)
        unsafe_positions.append(next_pos)

        # Add action to commands
        action = ship.move(direction)
        logging.info("ship: {}\nposition: {} -> {}\naction: {}\n, unsafes: {}\n".format(ship.id, ship.position, next_pos, action, unsafe_positions))
        commands.append(action)

        # Collect data
        data = [state, action_sample, ship.position, ship_data[ship.id][3], ship.halite_amount]
        ship_data[ship.id] = data
        
    if len(me.get_ships()) < 5 and not game_map[me.shipyard.position].is_occupied and me.halite_amount >= 1000 and shipyard not in unsafe_positions:
        commands.append(me.shipyard.spawn())

    game.end_turn(commands) 
    


