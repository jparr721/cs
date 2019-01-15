#!/usr/bin/env python3

import hlt
from hlt import constants
from hlt import player
from hlt.positionals import Direction, Position
import random
import copy
import logging


BAD_COORDINATES = [(1, 1), (1, -1), (-1, 1)]



def __init__(self):
    self.game = hlt.Game()
    self.game.read("ANNIHILATOR")
    self.game_map = self.game.game_map
    self.me = self.game.me
    self.direction = Direction()
    self.ships = self.me.get_ships()
    self.dropoff_count = 0
    self.ship_locations = {}
    logging.info('Bot created, ID is: {}.'.format(self.game.my_id))

def find_drop(self, array):
    a = 1000
    for val in array:
        b = game_map.calculate_distance(self, val)
        if b < a:
            a = b
    for val in array:
        b = game_map.calculate_distance(self, val)
        if b == a:
            return val

def adjust_ship_map(self, ship, x, y):
    self.ship_locations[ship.id] = (x, y)

def make_dropoff(self):
    if self.dropoff_count == 6:
        return
    # Max 6 dropoff points globalls
    self.dropoff_count += 1

def check_direction(self, coordinates):
    available_directions = []
    if coordinates not in BAD_COORDINATES:
        available_directions.append(self.direction.convert(coordinates))

def reload_ships(self):
    self.ships = self.me.get_ships()

def check_radius(self, ship, ship_locations):
    x, y = (ship.position.x, ship.position.y)
    for x_i in range(-1, 2):
        for y_i in range(-1, 2):
            new_direction = [x + x_i, y + y_i]
            cardinal_direction = (x_i, y_i)
            # IMPLEMENT DICT VALUE CHECK HERE

game = hlt.Game()
game.ready("MyPythonBot")
logging.info("Player ID is {}.".format(game.my_id))


def make_move(ship, move_vector): 
    tmp_vec = []
    for move in move_vector:
        tmp_vec = copy.deepcopy(move_vector)
        if game_map[ship.position.directional_offset(move)].is_occupied:
            tmp_vec.remove(move)
            
    if tmp_vec:
        if ship.halite_amount >= 50:
            ship_status[ship.id] = "returning"
            full_move = game_map.naive_navigate(ship, me.shipyard.position)
            logging.info("FULL MOVE: {}".format(full_move))
            return full_move
            
        return random.choice(tmp_vec)

    return Direction.Still
    
## SHIP LOCATIONS INDEXED BY {ship.id: [x,y]}
ship_locations = {}

def valid_moves(ship_pos, move_list, id):
    valids = [Direction.North, Direction.South, Direction.West, Direction.East]
    if not move_list:
        return valids

    z=0 
    temp_pos = []
    while z<4:
        temp_pos = ship_pos[id]
        if z==0:
            temp_pos[1] = temp_pos[1] - 1 
            temp_loc = Direction.North
        elif z==1:
            temp_pos[1] = temp_pos[1] + 1
            temp_loc = Direction.South
        elif z==2:
            temp_pos[0] = temp_pos[0] + 1
            temp_loc = Direction.East
        elif z==3:
            temp_pos[0] = temp_pos[0] - 1
            temp_loc = Direction.West
        
        logging.info("temppos: {}".format(temp_pos))
        logging.info("movelist: {}".format(move_list))

        
        if temp_pos in move_list:
            valids.remove(temp_loc)
            logging.info("Valids should be removed: {}".format(valids))
        elif temp_pos in ship_locations.values():
            logging.info("Ship locs: {}".format(ship_locations.values()))
            
            valids.remove(temp_loc)
        z = z + 1
    """
    logging.info("moves: {}".format(move_list))
    logging.info("ship position: {}".format(ship_pos[id]))
    logging.info("valids: {}".format(valids))
    """
    return valids
            
            
        

while True:
    game.update_frame()
    me = game.me
    game_map = game.game_map

    # A command queue holds all the commands you will run this turn. You build this list up and submit it at the
    #   end of the turn.
    command_queue = []
    ship_status = {}
    move_locations = []
    #move_vector = [Direction.North, Direction.South, Direction.West, Direction.East]

    for ship in me.get_ships():
        x, y = ship.position.x, ship.position.y
        ship_locations[ship.id] = [x,y]
        logging.info("Ship Locations: {}".format(ship_locations))
        logging.info("Ship {} has {} halite.".format(ship.id, ship.halite_amount))
        
        valid_move_list = valid_moves(ship_locations, move_locations, ship.id)

        if ship.id not in ship_status:
            ship_status[ship.id] = "exploring"


        if ship_status[ship.id] == "returning":
            if ship.position == me.shipyard.position:
                ship_status[ship.id] = "exploring"


        elif ship.halite_amount >= constants.MAX_HALITE / 4:
            ship_status[ship.id] = "returning"

        
        best_move = make_move(ship, valid_move_list)
        move_loc_temp = ship_locations[ship.id]
        if best_move == Direction.North:
            move_loc_temp[1] = move_loc_temp[1] - 1
            move_locations.append(move_loc_temp)
        elif best_move == Direction.West:
            move_loc_temp[0] = move_loc_temp[0] - 1
            move_locations.append(move_loc_temp)
        elif best_move == Direction.East:
            move_loc_temp[0] = move_loc_temp[0] + 1
            move_locations.append(move_loc_temp)
        elif best_move == Direction.South:
            move_loc_temp[1] = move_loc_temp[1] + 1
            move_locations.append(move_loc_temp)
        else:
            move_locations.append(ship_locations[ship.id])
        
        command_queue.append(ship.move(best_move))
        
    logging.info("Move list: {}".format(command_queue))
       

    # If the game is in the first 300 turns and you have enough halite, spawn a ship.
    # Don't spawn a ship if you currently have a ship at port, though - the ships will collide.
    if game.turn_number <= 300 and me.halite_amount >= constants.SHIP_COST and not game_map[me.shipyard].is_occupied:
        command_queue.append(me.shipyard.spawn())

    # Send your moves back to the game environment, ending this turn.
    game.end_turn(command_queue)