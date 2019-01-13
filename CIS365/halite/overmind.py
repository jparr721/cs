import hlt
from hlt import constants
from hlt.positions import Direction, Position
import random
import logging

class Individual(object):
    def __init__(self):
        self.fitness = -100
        self.genes = []
        self.gene_length = 8
        self.sight_distance = random.randint(1, 33)
        self.max_halite_move = random.randint(0, 1001)
        self.max_halite_return = random.randint(0, 1001)
        self.halite_so_far = 0

    def calculate_fitness(self):
        for i in range(self.gene_length):
            if self.genes[i] == 1:
                self.fitness += 1

class Population(object):
    def __inif__(self):
        self.population_size = 10
        self.inidividuals = [Individual() for _ in range(self.population_size)]
        self.fittest = 0

    def get_fittest():
        max_fitness = -29183710
        max_fitness_idx = 0

        for i in range(len(self.inidividuals)):
            indiv_fitness = self.inidividuals[i].fitness
            if max_fitness <= indiv_fitness:
                max_fitness = inidiv_fitness
                max_fitness_idx = i


class InterGalacticHaliteMuncher(object):
    def __init__(self):
        self.population = Population()
        self.fittest = None
        self.generation_count = 0

    def selection():
        fittest = self.population.get_fittest()
