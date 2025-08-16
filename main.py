
from env.grid_env import create_env

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
        if done and reward > 0:
            env.close()
            break