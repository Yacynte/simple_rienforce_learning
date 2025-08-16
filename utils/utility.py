# Imports:
# --------
import numpy as np

def _state_index(state : np.ndarray | list, grid_size: int = 5):
    state = np.array(state)
    if len(state.shape) > 1:
        index = []
        for st in state:
            index.append(grid_size*st[1] + st[0])
        index = np.array(index)
        return index
    else:
        return grid_size*state[1] + state[0]