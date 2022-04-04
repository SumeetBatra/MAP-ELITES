try:
    import Box2D

    from gym.envs.box2d.bipedal_walker import BipedalWalker, BipedalWalkerHardcore
except ImportError:
    Box2D = None
