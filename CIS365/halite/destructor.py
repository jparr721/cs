#!/usr/bin/env python3

import hlt
from hlt import constants
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

def find_drop(self, position, array):
    x = 1000
    for val in array:
        y = self.game_map.calculate_distance(position, array)
        if y < x:
            x = y
    return x

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
    valid_move = False 
    tmp_vec = []
    for move in move_vector:
        if not game_map[ship.position.directional_offset(move)].is_occupied:
            tmp_vec = copy.deepcopy(move_vector)
            tmp_vec.remove(move)
            valid_move = True
            
    if valid_move:
        return random.choice(tmp_vec)

    return Direction.Still

#        
#        if not game_map[ship.position.directional_offset(Direction.North)].is_occupied:
#            tmp_vec = copy.deepcopy(move_vector)
#            del tmp_vec(Direction.North)
#            return random.choice(tmp_vec)
#        elif not game_map[ship.position.directional_offset(Direction.South)].is_occupied:
#            return (ship.move(Direction.South))
#        elif not game_map[ship.position.directional_offset(Direction.West)].is_occupied:
#            return (ship.move(Direction.West))
#        elif not game_map[ship.position.directional_offset(Direction.East)].is_occupied:
#            return (ship.move(Direction.East))
#        else:
#            return (ship.stay_still())

    

while True:
    game.update_frame()
    me = game.me
    game_map = game.game_map

    # A command queue holds all the commands you will run this turn. You build this list up and submit it at the
    #   end of the turn.
    command_queue = []
    ship_status = {}
    ## SHIP LOCATIONS INDEXED BY [[x, y], ship_id]
    ship_locations = {}
    move_vector = [Direction.North, Direction.South, Direction.West, Direction.East]

    for ship in me.get_ships():
        x, y = ship.position.x, ship.position.y
        logging.info("Ship {} has {} halite.".format(ship.id, ship.halite_amount))

        if ship.id not in ship_status:
            ship_status[ship.id] = "exploring"

    ##This splits the ships into two random groups.  Group true searches east and false searches west.
#        if ship.group != True or ship.group != False:
#            z = random.randint(1,2)
#            if z == 1:
#                ship.group = True
#            else:
#                ship.group = False

        if ship_status[ship.id] == "returning":
            if ship.position == me.shipyard.position:
                ship_status[ship.id] = "exploring"
            #else:
                #move = game_map.naive_navigate(ship, find_drop(me.shipyard.position, me.get_dropoffs())
                #command_queue.append(ship.move(move))

        elif ship.halite_amount >= constants.MAX_HALITE / 4:
            ship_status[ship.id] = "returning"

        if ship.is_full:
            #move = game_map.naive_navigate(ship, find_drop(me.shipyard.position, me.get_dropoffs())
            #command_queue.append(ship.moveDirection(Direction.move))
            ship_status[ship_id] = "returning"
        
        best_move = make_move(ship, move_vector)
        logging.info('BEST MOVE: {}'.format(best_move))
        command_queue.append(ship.move(best_move))
#        if not game_map[ship.position.directional_offset(Direction.North)].is_occupied:
#            command_queue.append(ship.move(Direction.North))
#        elif not game_map[ship.position.directional_offset(Direction.South)].is_occupied:
#            command_queue.append(ship.move(Direction.South))
#        elif not game_map[ship.position.directional_offset(Direction.West)].is_occupied:
#            command_queue.append(ship.move(Direction.West))
#        elif not game_map[ship.position.directional_offset(Direction.East)].is_occupied:
#            command_queue.append(ship.move(Direction.East))
#        else:
#            command_queue.append(ship.stay_still())

       

    # If the game is in the first 300 turns and you have enough halite, spawn a ship.
    # Don't spawn a ship if you currently have a ship at port, though - the ships will collide.
    if game.turn_number <= 300 and me.halite_amount >= constants.SHIP_COST and not game_map[me.shipyard].is_occupied:
        command_queue.append(me.shipyard.spawn())

    # Send your moves back to the game environment, ending this turn.
    game.end_turn(command_queue)
