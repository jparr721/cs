#!/usr/bin/env python3

import hlt
from hlt import constants
from hlt import player
from hlt.positionals import Direction, Position
import random
import copy
import logging


BAD_COORDINATES = [(1, 1), (1, -1), (-1, 1)]
drop_locations = []
home_drop = True



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
    
def find_drop(ship):
    a = 1000
    for val in drop_locations:
        b = game_map.calculate_distance(ship, val)
        if b < a:
            a = b
    for val in drop_locations:
        b = game_map.calculate_distance(ship, val)
        if b == a:
            return val
        
def make_drop(ship):
    if len(drop_locations) > 3:
        return
    else:
        return True
        
def adjust_ship_map(self, ship, x, y):
    self.ship_locations[ship.id] = (x, y)

def make_dropoff(self, ship):
    if self.dropoff_count == 4:
        return
    # Max 4 dropoff points globalls
    
    if make_drop:
        drop_locations.append(ship.position)
        ship.make_dropoff()
    return

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
        if ship.halite_amount >= constants.MAX_HALITE * 0.8:
            ship_status[ship.id] = "returning"
            full_move = game_map.naive_navigate(ship, find_drop(ship.position))
            logging.info("FULL MOVE: {}".format(full_move))
            return full_move
            
        return random.choice(tmp_vec)

    return Direction.Still

## SHIP LOCATIONS INDEXED BY {ship.id: [x,y]}
ship_locations = {}

while True:
    game.update_frame()
    me = game.me
    game_map = game.game_map

    # A command queue holds all the commands you will run this turn. You build this list up and submit it at the
    #   end of the turn.
    command_queue = []
    ship_status = {}
    move_locations = []
    move_vector = [Direction.North, Direction.South, Direction.West, Direction.East]

    if home_drop:
        drop_locations.append(me.shipyard.position)
        home_drop = False
    
    for ship in me.get_ships():
        x, y = ship.position.x, ship.position.y
        ship_locations[ship.id] = [x,y]
        logging.info("Ship Locations: {}".format(ship_locations))
        logging.info("Ship {} has {} halite.".format(ship.id, ship.halite_amount))

        if ship.id not in ship_status:
            ship_status[ship.id] = "exploring"


        if ship_status[ship.id] == "returning":
            if ship.position == me.shipyard.position:
                ship_status[ship.id] = "exploring"


        elif ship.halite_amount >= constants.MAX_HALITE * 0.8:
            ship_status[ship.id] = "returning"


        best_move = make_move(ship, move_vector)

        command_queue.append(ship.move(best_move))

    logging.info("Move list: {}".format(command_queue))


    # If the game is in the first 300 turns and you have enough halite, spawn a ship.
    # Don't spawn a ship if you currently have a ship at port, though - the ships will collide.
    if game.turn_number % 6 == 0 and me.halite_amount >= constants.SHIP_COST and not game_map[me.shipyard].is_occupied:
        command_queue.append(me.shipyard.spawn())

    # Send your moves back to the game environment, ending this turn.
    game.end_turn(command_queue)
