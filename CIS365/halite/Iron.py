#!/usr/bin/env python3
import hlt
from hlt import constants
from hlt.positionals import Direction, Position
import logging

game = hlt.Game()
ship_locations = {}

# Saturation 2 is better though
OPTIMUM_HALITE_SATURATION = constants.MAX_HALITE / 2


def check_perimeter(game_map):
    spots = []
    # search all cells for halite
    # Possibly optimize by robot spot
    # for better performance
    for row in range(game_map.height):
        for col in range(game_map.width):
            pos = Position(row, col)
            if game_map[pos].halite_amount >= OPTIMUM_HALITE_SATURATION:
                spots.append(game_map[pos])

    # Get top 5 to keep things clean
    if spots > 5:
        return sorted(range(len(spots)), key=lambda i: spots[i])[-2:]
    else:
        return spots


def check_collision(ship, move, issa_map, ya_boy):
    move = Position(move)
    for ship_id in ship_locations:
        the_ship = ship_id['ship']
        the_next_move = ship_id['next_move']
        if the_next_move == move \
                and the_ship != ship:
            continue
        else:
            return False
    return True


def make_destination(ship, issa_map, ya_boy):
    ''' Finds the destination'''

    # If we don't have a lot of halite
    if ship.halite_amount >= OPTIMUM_HALITE_SATURATION * .2:
        ship_locations[ship.id]['move_to'] = Direction.Still

    possible_moves = issa_map.get_unsafe_moves(
            ship.position, ship_locations[ship.id]['moving_to'])

    for move in possible_moves:
        if not check_collision(ship, move, issa_map, ya_boy):
            ship_locations[ship.id] = Position(move)
            break
        else:
            continue
    # No valid moves? Stand still and suck some halite
    ship_locations[ship.id] = Direction.Still


def make_move(ship, ya_boy, issa_map):
    if ship.halite_amount >= constants.MAX_HALITE * 0.7:
        ship_locations[ship.id]['ship_status'] = 'returning'
        logging.info('Ship: {} going home'.format(ship.id))
        ship_locations[ship.id]['moving_to'] = ya_boy.shipyard.positon

    if ship_locations[ship.id]['ship_status'] == 'exploring':
        # dont leave if halite high
        if issa_map[ship.position].halite_amount >= OPTIMUM_HALITE_SATURATION:
            ship_locations[ship.id]['next_move'] = Direction.Still
        else:
            ship_locations[ship.id]['next_move'] = max(
                    check_perimeter(issa_map))


def check_ship_nearby(ship_id, ya_boy):
    return ship_locations[ship_id]['next_move'] != ya_boy.shipyard.position


game.ready('IRON')
logging.info('Iron has arrived... {}'.format(game.my_id))

while True:
    game.update_frame()
    ya_boy = game.me
    issa_map = game.game_map
    command_queue = []

    # Get the top 5 most halite spots
    most_halite = check_perimeter(issa_map, OPTIMUM_HALITE_SATURATION)

    # check dead boy
    for ship_id in ship_locations:
        if not ya_boy.has_ship(ship_id):
            del ship_locations[ship_id]

    # Check ship position
    for ship in ya_boy.get_ships():
        if ship.id not in ship_locations:
            ship_locations[ship.id] = {
                'ship': ship,
                'moving_to': (0, 0),
                'next_move': ship.postion,
                'ship_status': 'exploring'
            }
        else:
            make_destination(ship, issa_map, ya_boy)

    if game.turn_number <= 300 and \
        ya_boy.halite_amount >= constants.SHIP_COST \
            and not issa_map[ya_boy.shipyard].is_occupied:
        # Make sure no ships are nearby
        for ship_id in ship_locations:
            if check_ship_nearby(ship_id, ya_boy):
                command_queue.append(ya_boy.shipyard.spawn())

    for ship_id in ship_locations:
        command_queue.append(ship_id['ship'].move(ship_id['next_move']))

    game.end_turn(command_queue)
