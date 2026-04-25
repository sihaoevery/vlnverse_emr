from pydantic import BaseModel


class Config(BaseModel, extra='allow'):
    pass
