#main.py
"""Simple version of the NEAT"""

import neat
import random as rnd
import numpy as np
import math
import pygame as pg

HEIGHT = 800
WIDTH = 600

class Creature:
    def __init__(self, genome, config):
        x = rnd.randint(0,HEIGHT)
        y = rnd.randint(0,WIDTH)
        self.position = (x,y)
        self.angle = 0
        self.speed = 0
        self.fitness = 0
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)

    def get_inputs(self, food_position):
        dx = food_position[0] - self.position[0]
        dy = food_position[1] - self.position[1]
        distance_l2norm = np.linalg.norm([dx,dy])
        angle = math.atan2(dy, dx)
        return distance_l2norm, angle

    def update(self, food_position):
        inputs = self.get_inputs(food_position)
        output = self.net.activate(inputs)
        self.speed = output[0]
        self.angle = output[1]
        x = math.cos(self.angle) * self.speed
        y = math.sin(self.angle) * self.speed
        self.position = self.position[0] + x, self.position[1] + y

    def draw(self,surface):
        pg.draw.circle(surface, (255,255,255), (int(self.position[0]), int(self.position[1])), 10)






