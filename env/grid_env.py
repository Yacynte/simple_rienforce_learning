import gymnasium as gym
import pygame
import sys, math
import numpy as np

class OctagonEnv(gym.Env):
    def __init__(self, grid_size, start_coordinates, goal, wait):
        super(OctagonEnv, self).__init__()
        self.info = {}
        self.wait_time = wait
        self.start_coordinate = start_coordinates
        self.goal = np.array(goal)
        # self.agent_position = None
        self.grid_size = grid_size
        self.cell_size = 100
        self.danger_states = []
        self.done = False
        self.random_initialization = True

        # Define action and observation spaces:
        # ------------------------------------
        self.action_space = gym.spaces.Discrete(8)  # north, south, east, west, northwest, northeast, southwest, southeast
        self.agent_position = gym.spaces.Box(low=0, high=grid_size-1, shape=(2,), dtype=np.float32)
        
        # Display:
        # --------
        pygame.init()
        self.screen = pygame.display.set_mode((self.cell_size*self.grid_size, self.cell_size*self.grid_size))

        # Load and edit danger image:
        # -------------------------- 
        self.danger_image = pygame.image.load("rl_gridworld/img/danger.jpeg").convert()
        self.danger_image.set_colorkey((255, 255, 255))
        self.danger_image = pygame.transform.scale(self.danger_image, (self.cell_size, self.cell_size))

    def add_danger(self, coordinates):
        self.danger_states.append(np.array(coordinates))

    def reset(self):
        """
        Everything must be reset
        """
        if self.random_initialization:

            # Create a list of posible initialization positions:
            # -------------------------------------------------
            choices_x = [i for i in range(self.grid_size) if i not in self.danger_states[0]]
            choices_y = [i for i in range(self.grid_size) if i not in self.danger_states[1]]
            choices_y = [i for i in choices_y if i != self.goal[1]]
            choices_x = [i for i in choices_x if i != self.goal[0]]
            self.agent_position = np.array([np.random.choice(choices_x), np.random.choice(choices_y)])
        else:
            self.agent_position = np.array(self.start_coordinate)

        self.done = False
        self.reward = 0

        self.info["Distance to goal"] = np.sqrt(
            (self.agent_position[0]-self.goal[0])**2 +
            (self.agent_position[1]-self.goal[1])**2 )
        return self.agent_position, self.info

    
    def step(self, action):
        """
        Define agent movement logic
        """
        previous_position = np.copy(self.agent_position)
   	
        # North:
        # ----
        if action==0 and self.agent_position[0] > 0:
            self.agent_position[0] -= 1

        # South:
        # -----
        elif action==1 and self.agent_position[0] < self.grid_size -1 :
            self.agent_position[0] += 1

        # West:
        # -----
        elif action==2 and self.agent_position[1] > 0:
            self.agent_position[1] -= 1

        # East:
        # -----
        elif action==3 and self.agent_position[1] < self.grid_size -1:
            self.agent_position[1] += 1

        # Northwest:
        # ---------
        elif action==4 and np.all(self.agent_position > np.array((0,0))):
            self.agent_position -= np.array((1,1))

        # Northeast:
        # ---------
        elif action==5 and self.agent_position[0] > 0 and self.agent_position[1] < self.grid_size -1:
            self.agent_position[0] -= 1
            self.agent_position[1] += 1

        # Southwest:
        # ---------
        elif action==6 and self.agent_position[0] < self.grid_size -1 and self.agent_position[1] > 0:
            self.agent_position[0] += 1
            self.agent_position[1] -= 1

        # Southeast:
        # ---------
        elif action==7 and np.all(self.agent_position < np.array((self.grid_size -1, self.grid_size -1 ))):
            self.agent_position += np.array((1,1))

        # Check termination:
        # ------------------
        # Goal:
        if np.array_equal(self.agent_position, self.goal):
            self.done = True
            self.reward = +10
        # Danger:
        elif True in [np.array_equal(self.agent_position, each_danger) for each_danger in self.danger_states]:
            self.done = True
            self.reward = -20    
        else:
            self.done = False
            self.reward = -0.01
            if np.array_equal(np.abs(self.agent_position - previous_position), np.array((1,1))):
                self.reward = -0.015
            else:
                self.reward = -0.01

        

        # Info:
        # -----
        self.info["Distance to goal"] = np.sqrt(
            (self.agent_position[0]-self.goal[0])**2 + (self.agent_position[1]-self.goal[1])**2
            )

        return self.agent_position, self.done, self.reward, self.info


    # Draw Octagon shape:
    # ------------------
    def draw_octagon(self, surface, color, top_left, width=1, fill = False):
        cx = top_left[0] + self.cell_size // 2
        cy = top_left[1] + self.cell_size // 2
        r = self.cell_size // 2   # margin
        points = []
        for i in range(8):
            angle_deg = 45 * i - 45/2  # Rotate so the top is flat
            angle_rad = math.radians(angle_deg)
            x = cx + r * math.cos(angle_rad)
            y = cy + r * math.sin(angle_rad)
            points.append((x, y))
        if not fill:
            pygame.draw.polygon(surface, color, points, width)
        if fill:
            pygame.draw.polygon(surface, color, points)


    # Render environment :
    # ---------
    def render(self):
        # Close the window on quit
        for event in pygame.event.get():
            if event==pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Background:
        # -----------
        self.screen.fill((255,255,255))

        # Draw octagonal gridlines:
        # -------------------------
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                top_left = (col * self.cell_size, row * self.cell_size)
                self.draw_octagon(self.screen, (200, 200, 200), top_left, width=1)

                
        # Draw goal:
        # ---------
        goal = pygame.Rect(self.goal[1]*self.cell_size,
                    self.goal[0]*self.cell_size,
                    self.cell_size,
                    self.cell_size)
        self.draw_octagon(self.screen, (0, 255, 0), goal.topleft, fill=True)


        # Draw danger states:
        # ------------------
        for each_danger in self.danger_states:
            danger = pygame.Rect(each_danger[1]*self.cell_size,
                        each_danger[0]*self.cell_size,
                        self.cell_size,
                        self.cell_size)
            
            self.draw_octagon(self.screen, (200, 200, 200), danger.topleft, width=1)
            self.screen.blit(self.danger_image, danger.topleft)
            
        # Draw agent:
        # ----------
        agent = pygame.Rect(self.agent_position[1]*self.cell_size,
                    self.agent_position[0]*self.cell_size,
                    self.cell_size,
                    self.cell_size)
        self.draw_octagon(self.screen, (0, 0, 255), agent.topleft, fill=True)


        pygame.time.wait(self.wait_time)
        pygame.display.flip()


    # Method 4:
    # ---------
    def close(self):
        pygame.quit()


def create_env(danger_coordinates: np.ndarray | tuple | list, grid_size: int, 
               start_coordinates: np.ndarray | tuple | list, goal: np.ndarray | tuple | list, wait: int = 0):
    # Create environment:
    # -------------------
    env = OctagonEnv(grid_size, start_coordinates, goal, wait)
    for danger_ccordinate in danger_coordinates:
        env.add_danger(coordinates = danger_ccordinate)

    state, info = env.reset()
    env.render()
    print("Initial_state: ", state, "Distance to goal: ", info["Distance to goal"])
    return env


# Run as a script:
# ----------------
if __name__=="__main__":

    # Create environment:
    # -------------------

    grid_size = 5
    danger_coordinates = [[3,1],[2,4]]
    start_coordinate = [0,0]
    goal_coordinate = [grid_size-1, grid_size-1]

    for _ in range(1000):

        env = create_env(danger_coordinates, grid_size, start_coordinate, goal_coordinate, wait=100)

        for _ in range(150):

            next_step, done, reward, info = env.step(env.action_space.sample())
            env.render()
            print(f"Next-state: {next_step}, Done: {done}, Reward: {reward}, Distance to goal: {info['Distance to goal']}")

            # Restart Search if danger or goal is hit:
            # ---------------------------------------
            if done:
                env.close()
                break

        # Close Environment if the goal is attain:
        # ---------------------------------------
        # if done and reward > 0:
        #     env.close()
        #     break