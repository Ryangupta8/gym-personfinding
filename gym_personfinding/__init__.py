from gym.envs.registration import register

register(
    id='personfinding-v0',
    entry_point='gym_personfinding.envs:PersonFindingEnv',
)