from unilm.agents.base import BaseModel


class SummerizerAgent(BaseModel):
    def __init__(self):
        super().__init__(name="summerizer")
