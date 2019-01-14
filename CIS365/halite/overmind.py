import hlt
from hlt import constants
from hlt.positions import Direction, Position
import random
import logging


class IndividualHaliteMuncher(object):
    def __init__(self):
        self.halite = 0  # Halite represents our fitness
        self.sight_distance = random.randint(1, 33)
        self.max_halite_move = random.randint(0, 1001)
        self.max_halite_return = random.randint(0, 1001)
        self.ships_to_produce = random.randint(0, 101)
        self.stop_spawning = random.randint(0, 501)
        self.attribute_vector = [
                self.sight_distance,
                self.max_halite_move,
                self.max_halite_return,
                self.ships_to_produce,
                self.stop_spawning
                ]

    def calculate_fitness(self):
        '''
        The higher the halite, the higher the fitness, baby!
        '''
        return self.halite


class InterGalacticHaliteMuncher(object):
    def __init__(self, pop_size, generations):
        self.population = [IndividualHaliteMuncher()
                           for _ in range(self.pop_size)]
        self.fittest = None
        self.generation_count = 0

    def get_fittest(self, pop: list)->IndividualHaliteMuncher:
        best = None
        for individual in pop:
            if best is not None:
                best = max(best.calculate_fitness(),
                           individual.calculate_fitness())
            else:
                best = individual
        return best

    def selection(self, ratio: int = 2)->list:
        the_chosen_ones = []
        pop_tmp = self.population

        # Get amount based on ratio
        for _ in range(int(len(pop_tmp)/ratio)):
            local_fittest = self.get_fittest(pop_tmp)
            the_chosen_ones.append(local_fittest)
            pop_tmp.remove(local_fittest)

        return the_chosen_ones

    def mutation(self, mutation_point: int = int(5/3)):
        pass
