import numpy as np 
import pygame
import itertools
import math 

class Animat:
    
    def __init__(self, state):
        self.state = state
        self.num_actuators = 8
        self.last_action = None

        angle_step = (2 * np.pi) / self.num_actuators
        angle = 0.0
        self.actuator_effects = np.zeros( (self.num_actuators, 2) )
        self.actuator_angles = []
        for i in range(self.num_actuators):
            delta_x = np.sin( angle )
            delta_y = -np.cos( angle ) #For consistency with pygame positive y means heading down.

            self.actuator_effects[i, 0] = -delta_x
            self.actuator_effects[i, 1] = -delta_y
            self.actuator_angles.append( angle )
            angle += angle_step 

    
    def _action_effect(self, action):
        delta = np.zeros( (2,) )
        self.last_action = action
        for idx, a in enumerate(action):
            if a == 1:
                delta += self.actuator_effects[idx]
        return delta

    def step(self, action):
        delta = self._action_effect(action)
        
        new_state = self.state + delta 
        return new_state

    def render(self, scale, display):
        maze_state = (int(self.state[0]*scale), int(self.state[1]*scale))
        pygame.draw.circle(display, (255, 0, 0, 0), maze_state, scale, 1)
        if self.last_action is not None:
            for i in range(len(self.last_action)):
                a = self.last_action[i]
                delta_x = scale*np.sin(self.actuator_angles[i])
                delta_y = -scale*np.cos(self.actuator_angles[i])
                end_pos = (maze_state[0] + delta_x, maze_state[1] + delta_y)
                if a == 1:
                    pygame.draw.line(display, (0,255, 255, 0), maze_state, end_pos, 2)
                elif a == 0:
                    pygame.draw.line(display, (0, 0, 0, 0), maze_state, end_pos, 2)
       




class AnimatEnv:

    def __init__(self, maze_filename):
        lines = open(maze_filename, "r").readlines()

        self.maze = np.zeros((len(lines), len(lines[0])))
        self.animat = None
        self.display = None
        self.obstacle_locations = []
        self.env_range = np.array( [ [0, self.maze.shape[1]], [0, self.maze.shape[0]]] )
        self.scale = 20

        for row, line in enumerate(lines):
            for col, v in enumerate(line):
                if v == ".":
                    self.maze[row, col] = 1.0
                elif v == "a":
                    self.maze[row, col] = 1.0
                    self.initial_state = np.array([float(col), float(row)])
                elif v == "g":
                    self.maze[row, col] = 1.0
                    self.goal = (col, row)
                else:
                    self.obstacle_locations.append( (col, row) )


        self.reset()
        #### This is just to follow openai format ######
        class ActionSpace:
            def __init__(self, animat):
                self.n = 2**animat.num_actuators
                action_strings = itertools.product("01", repeat=animat.num_actuators)
                self.actions = []
                for s in action_strings:
                    a = [int(s[i]) for i in range(len(s))]
                    self.actions.append( a )
                

            def sample(self):
                return np.random.choice(range(len(self.actions)))
                

        self.action_space = ActionSpace(self.animat)



    def reset(self):
        self.animat = Animat(self.initial_state)
        return self.animat.state

    def step(self, action):
        new_state = self.animat.step( self.action_space.actions[action] )

        maze_state = (int(new_state[0]), int(new_state[1]))
        reward = -0.1
        if maze_state[0] > 0 and maze_state[0] < self.maze.shape[1] \
            and maze_state[1] > 0 and maze_state[1] < self.maze.shape[0] \
            and self.maze[maze_state[1], maze_state[0]] != 0:
            
            self.animat.state = new_state
        
        
        if  math.sqrt( (new_state[0]-self.goal[0])**2 + (new_state[1]-self.goal[1])**2 ) < 1.0:
            done = True
            reward = 10.0
        else:
            done = False 

        # print("State " + str(maze_state) + " Goal " + str(self.goal) + " Done " + str(done) + " Reward " + str(reward))

        return self.animat.state, reward, done, "Info"

    def render(self):
        if self.display is None:
            pygame.init()
            self.display = pygame.display.set_mode((self.maze.shape[1]*self.scale, self.maze.shape[0]*self.scale))
            fpsClock = pygame.time.Clock()
        
        self._update_screen()
        pygame.display.flip()



    def _update_screen(self):
        self.display.fill((255, 255, 255))
        self.animat.render(self.scale, self.display)

        for obstacle in self.obstacle_locations:
            obs = (obstacle[0]*self.scale, obstacle[1]*self.scale)
            pygame.draw.rect(self.display, (0, 0, 0, 0), (obs[0], obs[1], self.scale, self.scale) )


        goal_maze_state = (int(self.goal[0]*self.scale), int(self.goal[1]*self.scale))
        goal_radius = 5
        pygame.draw.circle(self.display, (255, 0, 0, 0), goal_maze_state, self.scale)

        
        
        