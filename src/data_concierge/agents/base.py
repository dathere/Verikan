"""Base agent class for all agents in the system."""

from abc import ABC, abstractmethod

from data_concierge.agents.state import GraphState
from data_concierge.core.logging import get_logger


class BaseAgent(ABC):
    """Abstract base class for all agents in the system."""

    name: str = "base_agent"
    description: str = "Base agent"

    def __init__(self) -> None:
        self.logger = get_logger(self.name)

    @abstractmethod
    async def process(self, state: GraphState) -> GraphState:
        """Process the current state and return updated state."""
        pass
