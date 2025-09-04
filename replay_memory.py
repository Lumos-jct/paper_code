from collections import deque
import random

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to memory.
        """
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Randomly sample a batch of experiences from memory.
        """
        if len(self.memory) < batch_size:
            return None
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)