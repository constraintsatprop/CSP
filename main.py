
import arcade
import pyglet.image  # what is this for?
import arcade
import numpy as np
import time
import copy
from collections import deque
from numpy import random, sin, cos
import os
import shutil
import tempfile

## adapted from the arcade examples

# Set how many rows and columns we will have
ROW_COUNT = 60
COLUMN_COUNT = 60

# This sets the margin between each cell
# and on the edges of the screen.
MARGIN = 0

ACTION_PROFILE = {0: [-1, 0], 1: [0, 1], 2: [0, -1], 3: [1, 0]}
ACTION_NAMES = dict(zip(["up", "left", "right", 'down'], range(4)))

# Do the math to figure out oiur screen dimensions
SCREEN_WIDTH = (600)
SCREEN_HEIGHT = (600)
WIDTH = int((SCREEN_WIDTH - MARGIN)/COLUMN_COUNT - MARGIN)
HEIGHT = int((SCREEN_HEIGHT - MARGIN)/ROW_COUNT - MARGIN)
SCREEN_WIDTH = (WIDTH + MARGIN) * COLUMN_COUNT + MARGIN
SCREEN_HEIGHT = (HEIGHT + MARGIN) * ROW_COUNT + MARGIN + 1  # the 1 is for a little extra padding.

N = COLUMN_COUNT*ROW_COUNT

class Gridworld(arcade.Window):
    def __init__(self, width, height):
        super().__init__(width, height)
        self.grid = np.zeros(shape=(ROW_COUNT, COLUMN_COUNT))
        arcade.set_background_color((200, 200, 200))
        self.draw_grid = False
        self.S_size = COLUMN_COUNT*ROW_COUNT
        self.S = self.grid
        self.omega = np.pi/300
        self.coord_frame_t = np.array([[SCREEN_WIDTH//2], [SCREEN_HEIGHT//2]])
        self.vector = self.coord_frame_t + np.array([[300],
                                                     [0]])  # x, y

    def render(self):
        arcade.start_render()
        # Draw the grid
        startx, starty = self.coord_frame_t.flatten().tolist()
        endx, endy = self.vector[[0,1]]
        endx2, endy2 = (-1*(self.vector-self.coord_frame_t) + self.coord_frame_t)[[0,1]]
        print(startx, starty, endx, endy)
        arcade.draw_line(startx, starty, endx, endy, color=arcade.color.DARK_GREEN, border_width = 30)
        arcade.draw_line(startx, starty, endx2, endy2, color=arcade.color.DARK_GREEN, border_width = 30)

        if self.draw_grid:
            for i in range(ROW_COUNT+1):
                y = i*HEIGHT
                arcade.draw_line(0, y, SCREEN_WIDTH, y, color=arcade.color.BLACK)
            for j in range(COLUMN_COUNT+1):
                x = j*WIDTH
                arcade.draw_line(x, 0, x, SCREEN_HEIGHT, color=arcade.color.BLACK)

    def on_draw(self):
        self.render()

    @staticmethod
    def get_xy_from_Cell_idx(row, column):
        x = (MARGIN + WIDTH) * column + MARGIN + WIDTH // 2
        y = (MARGIN + HEIGHT) * row + MARGIN + HEIGHT // 2
        return x, y

    def on_key_press(self, key, modifiers):
        if key == arcade.key.Q:
            self.end()
        elif key == arcade.key.ENTER:
            self.reset()

        elif key == arcade.key.SPACE:
            if self.do_reset:
                self.reset()
            self.cycle_run()

        elif key == arcade.key.RIGHT:
            self.speed = min(self.speed + 0.05, 1)
        elif key == arcade.key.LEFT:
            self.speed = max(self.speed - 0.05, 0.9)

    def cycle_run(self):
        self.stopped = not(self.stopped)

    def reset(self):
        self.do_reset = False
        self.s = np.array([0, 0])
        self.draw_cell_pdf = None

    def end(self):
        import sys
        sys.exit()

    def on_mouse_press(self, x: float, y: float, button: int, modifiers: int):
        column = x // WIDTH
        row = y // HEIGHT
        print(f"Click coordinates: ({x}, {y}). Grid coordinates: ({row}, {column})")


    @staticmethod
    def boundary_threshold(pos):
        if pos[0] >= COLUMN_COUNT:
            pos[0] = COLUMN_COUNT-1
        elif pos[0] < 0:
            pos[0] = 0
        if pos[1] >= ROW_COUNT:
            pos[1] = ROW_COUNT-1
        elif pos[1] < 0:
            pos[1] = 0
        return pos

    def update(self, delta_time = 1):
        # Update world state according to arbitrary function (differential equations, etc)

        # t = self.time
        # x = self.state  # state is a complicated variable containing all the components of the world we care aboue
        # theta = x.theta  # Current angle of windmill blade
        # omega = x.omega  # angular velocity (time-derivative of theta)
        omega = self.omega
        vector = self.vector - self.coord_frame_t
        R = np.matrix(
            [[cos(omega), -sin(omega)],
             [sin(omega), cos(omega)]])
        self.vector = np.matmul(R, vector).reshape(-1,1) + self.coord_frame_t


    def logical2coord(self, logical_index):
        #print('logical index', logical_index)
        y = logical_index // COLUMN_COUNT
        x = logical_index % COLUMN_COUNT
        #print('x and y', x, y)
        return np.array([x,y])

    def get_logical_index(self, coord):
        return int(coord[1]*COLUMN_COUNT + coord[0])



class Entity:
    def __init__(self, x, y, cost, color):
        coord = np.array([x, y])
        self.coord = coord
        self.cost = cost
        self.color = color
        self.scale = 1

    def set_coord(self, coord):
        self.coord = coord

    def set_color(self, color):
        self.color = color


class Agent(Entity):
    def __init__(self, x, y, cost=None, color=arcade.color.AZURE):
        super(Agent, self).__init__(x, y, cost, color)
        # self.scale = 1.25

class Reward(Entity):
    def __init__(self, x, y, cost=0, color=arcade.color.GREEN):
        super(Reward, self).__init__(x, y, cost, color)
        self.is_terminal = True
        # self.scale = 2

class Wall(Entity):
    def __init__(self, x, y, end_coord, cost, color=arcade.color.BROWN):
        super(Wall, self).__init__(x, y, cost, color)

def main():
    Gridworld(SCREEN_WIDTH, SCREEN_HEIGHT)
    arcade.run()

if __name__ == "__main__":
    main()

