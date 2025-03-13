import networkx as nx
import scipy.sparse.csgraph
import numpy as np
import gym
import pickle

WALLS = {
    'LavaPits': # 9 x 9 grid  x : 0..9, y : 0..9
        np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 1, 1, 1, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 1, 1, 1, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0],]),
}

ACT_DICT = {
    0: [0.,0.],
    1: [0., -1.],
    2: [0., 1.],
    3: [-1., 0.],
    4: [1., 0.],
}

def resize_walls(walls, factor):
  """Increase the environment by rescaling.
  
  Args:
    walls: 0/1 array indicating obstacle locations.
    factor: (int) factor by which to rescale the environment."""
  (height, width) = walls.shape
  row_indices = np.array([i for i in range(height) for _ in range(factor)])
  col_indices = np.array([i for i in range(width) for _ in range(factor)])
  walls = walls[row_indices]
  walls = walls[:, col_indices]
  assert walls.shape == (factor * height, factor * width)
  return walls

class DrunkSpider(gym.Env):
  """Abstract class for 2D navigation environments."""

  def __init__(self,
               dense_reward=False,
               ):
    
    """Initialize the drunk spider environment.

    Args:
      walls: (str) name of one of the maps defined above.
      resize_factor: (int) Scale the map by this factor.
      action_noise: (float) Standard deviation of noise to add to actions. Use 0
        to add no noise.
    """

    import matplotlib
    #matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    self.plt = plt
    self.fig = self.plt.figure()
    
    self.action_dim = self.ac_dim = 2
    self.observation_dim = self.obs_dim = 2
    self.env_name = 'drunkspider'
    self.is_gym = True

    resize_factor = 1
    walls = 'LavaPits'
    self.fixed_start = np.array([4.5, 8.5]) * resize_factor
    self.fixed_goal  = np.array([4.5, 0.5]) * resize_factor

    self.max_episode_steps = 30

    if resize_factor > 1:
      self._walls = resize_walls(WALLS[walls], resize_factor)
    else:
      self._walls = WALLS[walls]
    
    (height, width) = self._walls.shape
    self._apsp = self._compute_apsp(self._walls)

    self._height = height
    self._width = width
    self.action_space = gym.spaces.Discrete(5)
    self.observation_space = gym.spaces.Box(
        low=np.array([0,0]),
        high=np.array([self._height, self._width]),
        dtype=np.float64)

    self.dense_reward = dense_reward
    self.num_actions = 5
    self.epsilon = resize_factor
    self.action_noise = 0.1
    
    self.obs_vec = []
    self.last_trajectory = None

    self.num_runs = 0
    self.reset()

  def seed(self, seed):
    np.random.seed(seed)
    
  def reset(self, seed=None):

    if seed: 
      self.seed(seed)

    if len(self.obs_vec) > 0:
      self.last_trajectory = self.plot_trajectory()
    
    self.plt.clf()
    self.timesteps_left = self.max_episode_steps
    
    self.obs_vec = [self._normalize_obs(self.fixed_start.copy())]
    self.state = self.fixed_start.copy()
    self.num_runs += 1
    return self._normalize_obs(self.state.copy())

  def set_logdir(self, path):
    self.traj_filepath = path + 'last_traj.png'
    
  def _get_distance(self, obs, goal):
    """Compute the shortest path distance.
    
    Note: This distance is *not* used for training."""
    (i1, j1) = self._discretize_state(obs.copy())
    (i2, j2) = self._discretize_state(goal.copy())
    return self._apsp[i1, j1, i2, j2]

  def simulate_step(self, state, action):
    num_substeps = 10
    dt = 1.0 / num_substeps
    num_axis = len(action)
    for _ in np.linspace(0, 1, num_substeps):
      for axis in range(num_axis):
        new_state = state.copy()
        new_state[axis] += dt * action[axis]

        if not self._is_blocked(new_state):
          state = new_state

    return state

  def get_optimal_action(self, state):
    #state = self._unnormalize_obs(state)
    best_action = 0
    best_dist = np.inf
    for i in range(self.num_actions):
      action = np.array(ACT_DICT[i])
      s_prime = self.simulate_step(state, action)
      dist = self._get_distance(s_prime, self.fixed_goal)
      if dist < best_dist:
        best_dist = dist
        best_action = i
    return best_action

  def _discretize_state(self, state, resolution=1.0):
    (i, j) = np.floor(resolution * state).astype(np.int)
    # Round down to the nearest cell if at the boundary.
    if i == self._height:
      i -= 1
    if j == self._width:
      j -= 1
    return (i, j)

  def _normalize_obs(self, obs):
    return np.array([
      obs[0] / float(self._height),
      obs[1] / float(self._width)
    ])

  def _unnormalize_obs(self, obs):
    return np.array([
      obs[0] * float(self._height),
      obs[1] * float(self._width)
    ])
  
  def _is_blocked(self, state):
    return not self.observation_space.contains(state)

  def _is_dead(self, state):
    (i, j) = self._discretize_state(state)

    if self._is_blocked:
      return False
    
    return (self._walls[i, j] == 1)

  def step(self, action):
    self.timesteps_left -= 1

    if isinstance(action, np.ndarray):
      action = action.item()

    action = np.array(ACT_DICT[action])
    action = np.random.normal(action, self.action_noise)
    self.state = self.simulate_step(self.state, action)

    is_dead = self._is_dead(self.state)

    dist = np.linalg.norm(self.state - self.fixed_goal)
    done = ((dist < self.epsilon) or (self.timesteps_left == 0)) or is_dead
    ns = self._normalize_obs(self.state.copy())
    self.obs_vec.append(ns.copy())
    
    if self.dense_reward:
      reward = -dist
    else:
      reward = int(dist < self.epsilon) - 1

    if is_dead:
      reward = -100

    return ns, reward, done, {}

  @property
  def walls(self):
    return self._walls

  @property
  def goal(self):
    return self._normalize_obs(self.fixed_goal.copy())

  def _compute_apsp(self, walls):
    (height, width) = walls.shape
    g = nx.Graph()
    # Add all the nodes
    for i in range(height):
      for j in range(width):
        if walls[i, j] == 0:
          g.add_node((i, j))

    # Add all the edges
    for i in range(height):
      for j in range(width):
        for di in [-1, 0, 1]:
          for dj in [-1, 0, 1]:
            if di == dj == 0: continue  # Don't add self loops
            if i + di < 0 or i + di > height - 1: continue  # No cell here
            if j + dj < 0 or j + dj > width - 1: continue  # No cell here
            if walls[i, j] == 1: continue  # Don't add edges to walls
            if walls[i + di, j + dj] == 1: continue  # Don't add edges to walls
            g.add_edge((i, j), (i + di, j + dj))

    # dist[i, j, k, l] is path from (i, j) -> (k, l)
    dist = np.full((height, width, height, width), np.float('inf'))
    for ((i1, j1), dist_dict) in nx.shortest_path_length(g):
      for ((i2, j2), d) in dist_dict.items():
        dist[i1, j1, i2, j2] = d

    return dist

  def render(self, mode=None):
    self.plot_walls()

    # current and end
    self.plt.plot(self.fixed_goal[0], self.fixed_goal[1], 'go')
    self.plt.plot(self.state[0], self.state[1], 'ko')
    self.plt.pause(0.1)

    self.fig.canvas.draw()
    img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
    return img

  def plot_trajectory(self):

    self.plt.clf()
    self.plot_walls()

    obs_vec, goal = np.array(self.obs_vec), self.goal
    self.plt.plot(obs_vec[:, 0], obs_vec[:, 1], 'b-o', alpha=0.3)
    self.plt.scatter([obs_vec[0, 0]], [obs_vec[0, 1]], marker='X',
                color='red', s=150, label='start', alpha=0.7)
    self.plt.scatter([obs_vec[-1, 0]], [obs_vec[-1, 1]], marker='D',
                color='lightblue', s=130, label='end', alpha=0.7)
    self.plt.scatter([goal[0]], [goal[1]], marker='o',
                color='lightgreen', s=200, label='goal', alpha=0.7)
    self.plt.legend(loc='upper left')
    self.plt.savefig(self.traj_filepath)

  def get_last_trajectory(self):
    return self.last_trajectory

  def plot_walls(self, walls=None):
    if walls is None:
      walls = self._walls.T
    (height, width) = walls.shape
    for (i, j) in zip(*np.where(walls)):
      x = np.array([j, j+1]) / float(width)
      y0 = np.array([i, i]) / float(height)
      y1 = np.array([i+1, i+1]) / float(height)
      self.plt.fill_between(x, y0, y1, color='darkred')
    self.plt.xlim([0, 1])
    self.plt.ylim([0, 1])
    self.plt.xticks([])
    self.plt.yticks([])
  
  def _sample_normalized_empty_state(self):
    s = self._sample_empty_state()
    return self._normalize_obs(s)
  
  def _sample_empty_state(self):
    candidate_states = np.where(self._walls == 0)
    num_candidate_states = len(candidate_states[0])
    state_index = np.random.choice(num_candidate_states)
    state = np.array([candidate_states[0][state_index],
                      candidate_states[1][state_index]],
                     dtype=np.float)
    state += np.random.uniform(size=2)
    assert not self._is_blocked(state)
    assert not self._is_dead(state)
    return state

def refresh_path():
  path = dict()
  path['observations'] = []
  path['actions'] = []
  path['next_observations'] = []
  path['terminals'] = []
  path['rewards'] = []
  return path

if __name__ == '__main__':
  env = DrunkSpider(dense_reward=False)
  env.set_logdir('./')
  seed = np.random.randint(1000)
  num_samples = 50000
  total_samples = 0

  path = refresh_path()
  all_paths = []
  num_positive_rewards = 0
  
  while total_samples < num_samples:

    path = refresh_path()  
    bern = None#(np.random.rand() > 1.0)

    if bern:
      goal_state = env._sample_empty_state()
    else:
      goal_state = env.fixed_goal

    # curr_state = start_state
    curr_state = env.reset(seed)
    print ('Start: ', curr_state, ' Goal state: ', goal_state, total_samples)

    done = False
    for i in range(env.max_episode_steps):
      
      action = env.get_optimal_action(goal_state)
      temp_bern = (np.random.rand() < 0.2)

      if temp_bern:
        action = np.random.randint(5)
      next_state, reward, done, _ = env.step(action)

      if reward >= 0:
        num_positive_rewards += 1
      path['observations'].append(curr_state)
      path['actions'].append(action)
      path['next_observations'].append(next_state)
      path['terminals'].append(done)
      path['rewards'].append(reward)

      if done == True:
        total_samples += i
        break
    env.render()
    all_paths.append(path)
    print ('Num Positive Rewards: ', num_positive_rewards)

  with open('buffer_debug_final' + str(env.difficulty) +'.pkl', 'wb') as f:
    pickle.dump(all_paths, f)
