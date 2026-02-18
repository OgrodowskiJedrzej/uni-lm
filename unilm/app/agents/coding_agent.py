from .base import BaseModel

class CodingAgent(BaseModel):
    def __init__(self):
        super().__init__(name = "coder")