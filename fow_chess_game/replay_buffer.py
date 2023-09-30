import random
from collections import namedtuple, deque

Experience = namedtuple("Experience", ("state", "value"))


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params:
        -------
        buffer_size: int
            maximum size of buffer
        batch_size: int
            size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, value):
        """Add a new experience to memory."""
        experience = Experience(state, value)
        self.memory.append(experience)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
