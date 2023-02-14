from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env
import sys
import os
import hydra
from omegaconf import DictConfig
from .env.pursuit_ma_env import PursuitMAEnv
import yaml

FLAG = True

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}


REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)


with open(os.path.join(os.path.dirname(__file__), "env", "{}.yaml".format('pursuit_ma_env')), "r") as f:
    try:
        config_dict = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        assert False, "{}.yaml error: {}".format('pursuit_ma_env', exc)
env = PursuitMAEnv(config_dict)
# REGISTRY["PE"] = partial(env_fn, env=env)
REGISTRY["PE"] = env


if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))

