import gym
from gym import error, spaces, utils
from gym.utils import seeding

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

"""
    Grid World environment from Sutton's Reinforcement Learning book chapter 4.
    You are an agent on an MxN grid and your goal is to reach the terminal
    state at the top left or the bottom right corner.
    For example, a 4x4 grid looks as follows:
    T  o  o  o
    o  x  o  o
    o  o  o  o
    o  o  o  T
    x is your position and T are the two terminal states.
    You can take actions in each direction (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions going off the edge leave you in your current state.
    You receive a reward of -1 at each step until you reach a terminal state.
    """

class PersonFindingEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self, shape=[5,5]):
	self.shape = shape

	nS = np.prod(shape) # number of states
	nA = 4 # number of actions

	MAX_Y = shape[0]
	MAX_X = shape[1]

	P = {}
	grid = np.arange(nS).reshape(shape)
	it = np.nditer(grid, flags=['multi_index'])

	while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            # P[s][a] = (prob, next_state, reward, is_done)
            P[s] = {a : [] for a in range(nA)}

            is_done = lambda s: s == 1 or s == 5 or s == 6 # we finish when we enter a square next to P1 (Target)
            reward = 1000. if is_done(s) else -1.0

            # We're stuck in a terminal state
            if is_done(s):
                P[s][UP] = [(1.0, s, reward, True)]
                P[s][RIGHT] = [(1.0, s, reward, True)]
                P[s][DOWN] = [(1.0, s, reward, True)]
                P[s][LEFT] = [(1.0, s, reward, True)]
            # Not a terminal state
            else:
                ns_up = s if y == 0 else s - MAX_X
                ns_right = s if x == (MAX_X - 1) else s + 1
                ns_down = s if y == (MAX_Y - 1) else s + MAX_X
                ns_left = s if x == 0 else s - 1
                P[s][UP] = [(1.0, ns_up, reward, is_done(ns_up))]
                P[s][RIGHT] = [(1.0, ns_right, reward, is_done(ns_right))]
                P[s][DOWN] = [(1.0, ns_down, reward, is_done(ns_down))]
                P[s][LEFT] = [(1.0, ns_left, reward, is_done(ns_left))]

            it.iternext()

        # Initial state distribution is uniform
        isd = np.ones(nS) / nS

        # We expose the model of the environment for educational purposes
        # This should not be used in any model-free learning algorithm
        self.P = P

        super(GridworldEnv, self).__init__(nS, nA, P, isd)

  def render(self, mode='human', close=False):
    """ Renders the current gridworld layout
         For example, a 5x5 grid with the mode="human" looks like:
            T  o  o  o
            o  x  o  o
            o  o  o  o
            o  o  o  T
        where x is your position and T are the two terminal states.
        """
		if close:
			return

		outfile = io.StringIO() if mode == 'ansi' else sys.stdout

		grid = np.arange(self.nS).reshape(self.shape)
		it = np.nditer(grid, flags=['multi_index'])
		while not it.finished:
			s = it.iterindex
			y, x = it.multi_index

			if self.s == s:
				output = " x "
			elif s == 0:
				output = " P1 "
			elif s == 5-1:
				output = " P2 "
			elif s == 21-1:
				output = " P3 "
			else:
				output = " o "

			if x == 0:
				output = output.lstrip()
			if x == self.shape[1] - 1:
				output = output.rstrip()

            outfile.write(output)

            if x == self.shape[1] - 1:
                outfile.write("\n")

            it.iternext()

	def step(self, action):
		obs, reward, done, info = super().step(action)

		if action == self.actions.pickup:
			if self.carrying and self.carrying == self.obj:
				reward = self._reward()
				done = True

	def reset(self):
		# Current position and direction of the agent
		self.agent_pos = None

		# Generate a new random grid at the start of each episode
		# To keep the same grid for each episode, call env.seed() with
		# the same seed before calling env.reset()
		self._gen_grid(self.width, self.height)

		# These fields should be defined by _gen_grid
		assert self.agent_pos is not None

		# Check that the agent doesn't overlap with an object
		start_cell = self.grid.get(*self.agent_pos)
		assert start_cell is None or start_cell.can_overlap()

		# Item picked up, being carried, initially nothing
		self.carrying = None

		# Step count since episode start
		self.step_count = 0

		# Return first observation
		obs = self.gen_obs()
		return obs



		