from agents.base import BaseModel


class ReviewerAgent(BaseModel):
    def __init__(self):
        super().__init__(name="reviewer")
