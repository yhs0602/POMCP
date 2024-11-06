import abc
import random
from typing import TypeVar, List

Observation = TypeVar("Observation")


class State(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def is_terminal(self) -> bool:
        # Return True if the state is terminal
        pass

    @abc.abstractmethod
    def sample_random_action(self):
        # Return a random action
        pass


class Model(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def __call__(self, state: State, action) -> (State, Observation, float):
        # Given a state and an action, return the next state, observation and reward
        pass

    @abc.abstractmethod
    def is_consistent(self, state: State, observation: Observation) -> bool:
        # Return True if the observation is consistent with the state
        pass


class Node:
    def __init__(self):
        self.children = {}  # {action: Node}
        self.visit_count = 0
        self.value = 0
        self.particles: List[State] = []

    def add_particle(self, particle: State):
        self.particles.append(particle)

    def populate_particles(self, observation: Observation, model: Model):
        # Populate particles from the observation
        # You might want to use this function to save memory
        pass

    def sample_particle(self) -> State:
        # Randomly sample a particle from this node's belief
        return random.choice(self.particles)

    def update(self, reward):
        self.value = (self.value * self.visit_count + reward) / (self.visit_count + 1)
        self.visit_count += 1

    def prune_particles(self, observation: Observation, model: Model):
        # Remove particles that are not consistent with the observation
        self.particles = [p for p in self.particles if model.is_consistent(p, observation)]


class POMCP2:
    def __init__(
            self,
            model,
            initial_belief: State,
            max_depth=30,
            gamma=0.99,
            num_simulations=1000,
    ):
        self.root = Node()
        self.root.add_particle(initial_belief)
        self.model = model
        self.max_depth = max_depth
        self.gamma = gamma
        self.num_simulations = num_simulations

    def search(self, observation: Observation, action_mask) -> int:
        # prune the beliefs that are not reachable from the current observation
        self.root.prune_particles(observation, self.model)

        # run simulations to estimate the value of each action
        for _ in range(self.num_simulations):
            particle = self.root.sample_particle()
            self._simulate(particle, self.root, 0, action_mask)
        # Return action with the highest value from root's children
        return max(self.root.children, key=lambda a: self.root.children[a].value)

    # Must be called after action to update the root node
    def prune_after_action(self, action):
        # Prune the tree after taking an action
        # The root node should be the child of the current root node
        # You might want to use this function to save memory
        child = self.root.children[action]
        del self.root
        self.root = child

    def _simulate(self, particle: State, node: Node, depth: int, action_mask):
        if depth >= self.max_depth:
            return 0

        valid_actions = [a for a in node.children if action_mask[a]]
        if not valid_actions:
            # No valid actions to take, end the simulation
            return 0

        # UCB1 to select action, you might want to add the UCB1 formula here
        action = self._select_action_ucb1(node, valid_actions)

        # Expand the tree if the action is not in the tree
        if action not in node.children:
            node.children[action] = Node()
            return self.rollout(particle)

        new_particle, new_observation, reward = self.model(particle, action)
        child_node = node.children[action]
        child_node.add_particle(new_observation)

        q_value = reward + self.gamma * self._simulate(
            new_particle, child_node, depth + 1, action_mask
        )
        child_node.update(q_value)

        return q_value

    def _select_action_ucb1(self, node, valid_actions):
        # Implement the UCB1 formula to select action based on exploration vs exploitation
        # For simplicity, we'll use random choice among valid actions in this placeholder
        return random.choice(valid_actions)

    def rollout(self, particle: State):
        # Implement your default policy here, e.g., random actions until end of episode or max depth
        total_reward = 0
        while not particle.is_terminal() and depth < self.rollout_depth:
            action = particle.sample_random_action()
            new_particle, _, reward = self.model(particle, action)
            total_reward += reward
            particle = new_particle
        return total_reward
