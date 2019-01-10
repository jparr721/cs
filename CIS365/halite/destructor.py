#!/usr/bin/env python3

import hlt
from hlt import constants
from hlt.positions import Direction, Position
import random
import logging


BAD_COORDINATES = [(1, 1), (1, -1), (-1, 1)]


class Annihilator(object):
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


while True:
    BUTTHOLE_SHREDDER = Annihilator()
