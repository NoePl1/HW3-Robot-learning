from hw8.roble.envs import ant
from hw8.roble.envs import cheetah
from hw8.roble.envs import obstacles
from hw8.roble.envs import reacher
from hw8.roble.envs import pointmass  # 🚀 Add this line

from gym.envs.registration import register

def register_envs():
    register(
        id='cheetah-roble-v0',
        entry_point='roble.envs.cheetah:HalfCheetahEnv',
        max_episode_steps=1000,
    )
    register(
        id='ant-roble-v0',
        entry_point='roble.envs.ant:AntEnv',
        max_episode_steps=1000,
    )
    register(
        id='obstacles-roble-v0',
        entry_point='roble.envs.obstacles:Obstacles',
        max_episode_steps=500,
    )
    register(
        id='reacher-roble-v0',
        entry_point='roble.envs.reacher:Reacher7DOFEnv',
        max_episode_steps=500,
    )
    # ✅ Register Pointmass Environments
    register(
        id="PointmassEasy-v0",
        entry_point="hw8.roble.envs.pointmass:PointmassEasyEnv",
        max_episode_steps=200,
    )
    register(
        id="PointmassMedium-v0",
        entry_point="hw8.roble.envs.pointmass:PointmassMediumEnv",
        max_episode_steps=200,
    )
    register(
        id="PointmassHard-v0",
        entry_point="hw8.roble.envs.pointmass:PointmassHardEnv",  # ✅ Fix this path if needed
        max_episode_steps=200,
    )
    register(
        id="PointmassVeryHard-v0",
        entry_point="hw8.roble.envs.pointmass:PointmassVeryHardEnv",
        max_episode_steps=200,
    )
    register(
        id="DrunkSpider-v0",
        entry_point="hw8.roble.envs.pointmass:DrunkSpiderEnv",
        max_episode_steps=200,
    )