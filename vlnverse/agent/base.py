from typing import Any, Dict

from vlnverse.configs.agent import AgentCfg


class Agent:
    agents = {}

    def __init__(self, config: AgentCfg):
        self.config = config

    def step(self, obs: Dict[str, Any]):
        pass

    def reset(self):
        pass

    @classmethod
    def register(cls, agent_type: str):
        """
        Register a agent class.
        """

        def decorator(agent_class):
            cls.agents[agent_type] = agent_class

        return decorator

    @classmethod
    def init(cls, config: AgentCfg):
        """
        Init a agent instance from a config.
        """
        return cls.agents[config.model_name](config)
