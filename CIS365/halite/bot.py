#!/usr/bin/env python3
# Python 3.6

### MAX 6 DROPOFFS
### FAVOR HIGH-HALITE ZONES

# Import the Halite SDK, which will let you interact with the game.
import hlt

# This library contains constant values.
from hlt import constants

# This library contains direction metadata to better interface with the game.
from hlt.positionals import Direction

# This library allows you to generate random numbers.
import random

# Logging allows you to save messages for yourself. This is required because the regular STDOUT
#   (print statements) are reserved for the engine-bot communication.
import logging

""" <<<Game Begin>>> """

# This game object contains the initial game state.
game = hlt.Game()
# At this point "game" variable is populated with initial map data.
# This is a good place to do computationally expensive start-up pre-processing.
# As soon as you call "ready" function below, the 2 second per turn timer will start.
game.ready("rekbot")

# Now that your bot is initialized, save a message to yourself in the log file with some important information.
#   Here, you log here your id, which you can always fetch from the game object by using my_id.
logging.info("Successfully created bot! My Player ID is {}.".format(game.my_id))


""" Finds the optimal dropoff for a ship """
def findDrop( position, array ):
    x = 1000
    for val in array:
        y = game_map.calculate_distance(position, array)
        if y < x:
            x = y
    return x

def add_ship_to_map(ship, ship_locations, x, y):
    map[[x, y]] = ship.id

def make_dropoff():
    # MAX 6 DROPOFFS
    pass

def check_direction(coordinates: list):
     ### MAKE SURE COORDINATES ARE NEVER [1, 1] [1, -1], [-1, 1]
     ## RETURN THE VALID CARDINAL DIRECTIONS TO MOVE

def check_radius(ship, ship_locations):
    x, y = ship.position.x, ship.position.y
    for x_i in range(-1, 1):
#        if x + x_i < 0:
#            x_i = x
        for y_i in range(-1, 1):
            ## CHECK AROUND SHIP HERE
            new_dir = [x + x_i, y + y_i]
            cardinal_direction = (x_i, y_i)
            if new_dir in ship_locations:
                check_direction(new_dir)
            ## USE IS OCCUPIED TO CHECK

""" <<<Game Loop>>> """

while True:
    # This loop handles each turn of the game. The game object changes every turn, and you refresh that state by
    #   running update_frame().
    game.update_frame()
    # You extract player metadata and the updated map metadata here for convenience.
    me = game.me
    game_map = game.game_map

    # A command queue holds all the commands you will run this turn. You build this list up and submit it at the
    #   end of the turn.
    command_queue = []
    ship_status = {}
    ## SHIP LOCATIONS INDEXED BY [[x, y], ship_id]
    ship_locations = {}

    for ship in me.get_ships():
        x, y = ship.position.x, ship.position.y
        logging.info("Ship {} has {} halite.".format(ship.id, ship.halite_amount))

        if ship.id not in ship_status:
            ship_status[ship.id] = "exploring"

        if ship_status[ship.id] == "returning":
            if ship.position == me.shipyard.position:
                ship_status[ship.id] = "exploring"
            else:
                move = game_map.naive_navigate(ship, findDrop(me.shipyard.position, me.get_dropoffs())
                comand_queue.append(ship.move(move))

        elif ship.halite_amount >= constants.MAX_HALITE / 4:
            ship_status[ship.id] = "returning"

        if game_map[ship.position].halite_amount < constants.MAX_HALITE / 10 or ship.is_full:
            #for i in range(-1, 1):
                #for j in range(-1, 1):
#            if game_map[ship.position].is_occupied:
#                if i == 1:
#                    command_queue.append(ship.move(random.choice([Direction.South, Direction.West])))
#                if i == -1:
#                    command_queue.append(ship.move(random.choice([Direction.North, Direction.East])))
            command_queue.append(ship.move(random.choice([Direction.South, Direction.West, Direction.North, Direction.East])))
        else:
            command_queue.append(ship.stay_still())

    # If the game is in the first 200 turns and you have enough halite, spawn a ship.
    # Don't spawn a ship if you currently have a ship at port, though - the ships will collide.
    if game.turn_number <= 300 and me.halite_amount >= constants.SHIP_COST and not game_map[me.shipyard].is_occupied:
        command_queue.append(me.shipyard.spawn())

    # Send your moves back to the game environment, ending this turn.
    game.end_turn(command_queue)


ship.position.directional_offset(test_direction)
position.get_surrounding_cardinals()
game_map[position].halite_amount
