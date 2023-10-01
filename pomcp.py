import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from numpy.random import choice

from auxilliary import BuildTree, UCB
from fow_chess_game.fow_state import FowState
from fow_chess_game.replay_buffer import ReplayBuffer


# POMCP solver
class POMCP:
    # gamma = discount rate
    # c = higher value to encourage UCB exploration
    # threshold = threshold below which discount is too little
    # timeout = number of runs from node
    def __init__(
        self,
        generator,
        device,
        value_network,
        gamma=0.95,
        c=1,
        threshold=0.005,
        timeout=10000,
        no_particles=1200,
        buffer_size=100000,
        batch_size=32,
    ):
        self.gamma = gamma
        if gamma >= 1:
            raise ValueError("gamma should be less than 1.")
        self.Generator = generator
        self.e = threshold
        self.c = c
        self.timeout = timeout
        self.no_particles = no_particles
        self.tree = BuildTree()
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(
            buffer_size=buffer_size, batch_size=batch_size
        )
        self.value_network = value_network
        self.device = device
        self.optimizer = optim.Adam(self.value_network.parameters(), lr=0.001)

    # give state, action, and observation space
    def initialize(self, state, action_space, observation, action_mask):
        self.state = state
        self.actions = action_space
        self.observations = observation
        self.action_mask = action_mask

    # searchBest action to take
    # UseUCB = False to pick best value at end of Search()
    def SearchBest(self, h, state, use_ucb=True):
        max_value = None
        result = None
        resulta = None
        action_mask = self.action_mask(state)
        valid_actions = [
            action for i, action in enumerate(self.actions) if action_mask[i]
        ]
        if use_ucb:
            if self.tree.nodes[h].belief_particles != -1:
                children = self.tree.nodes[h].children
                # UCB for each child node
                for action, child in children.items():
                    # if node is unvisited return it
                    if (
                        self.tree.nodes[child].visit_count == 0
                        and action in valid_actions
                    ):
                        return action, child
                    ucb = UCB(
                        self.tree.nodes[h].visit_count,
                        self.tree.nodes[child].visit_count,
                        self.tree.nodes[child].value,
                        self.c,
                    )

                    # Max is kept
                    if (
                        max_value is None or max_value < ucb
                    ) and action in valid_actions:
                        max_value = ucb
                        result = child
                        resulta = action
            # return action-child_id values
            return resulta, result
        else:
            if self.tree.nodes[h].belief_particles != -1:
                children = self.tree.nodes[h].children
                # pick optimal value node for termination
                for action, child in children.items():
                    node_value = self.tree.nodes[child].value
                    # keep max
                    if (
                        max_value is None or max_value < node_value
                    ) and action in valid_actions:
                        max_value = node_value
                        result = child
                        resulta = action
            return resulta, result

    # Search module
    def Search(self):
        Bh = self.tree.nodes[-1].belief_particles.copy()
        print(f"Search {Bh=}")
        # Repeat Simulations until timeout
        for _ in range(self.timeout):
            if Bh == []:
                state = self.state  # choice(self.states)
            else:
                state = choice(Bh)
            self.Simulate(state, -1, 0)
        # Get best action
        action, _ = self.SearchBest(-1, state, use_ucb=False)
        return action

    # Check if a given observation node has been visited
    def getObservationNode(self, h, sample_observation):
        if sample_observation not in list(self.tree.nodes[h].children.keys()):
            # If not create the node
            self.tree.expand_tree_by_one_node(h, sample_observation)
        # Get the nodes index
        next_node = self.tree.nodes[h].children[sample_observation]
        return next_node

    def rollout(self, state: FowState, depth):
        # Check significance of update
        if (self.gamma**depth < self.e or self.gamma == 0) and depth != 0:
            return 0

        # If we've reached a certain depth, estimate the value using the value network instead of rolling out further.
        if depth >= 5:  # You can adjust this depth value as you see fit
            with torch.no_grad():
                state_tensor = (
                    torch.FloatTensor(state.array).unsqueeze(0).to(self.device)
                )
                # Assuming your state s can be converted directly to tensor and device is your target device (e.g., 'cuda')
                value_estimate = self.value_network(state_tensor)
                # Assuming value_network is an attribute of the POMCP class
            return value_estimate.item()

        initial_state = state  # Store the initial state
        cum_reward = 0

        # Pick random action; maybe change this later
        # Need to also add observation in history if this is changed
        action_mask = self.action_mask(state)
        valid_actions = np.where(action_mask == 1)[0]

        action_idx = np.random.choice(valid_actions)
        action = self.actions[action_idx]
        # print(f"{valid_actions=} {action=}")
        # action = choice(self.actions)

        # Generate states and observations
        # print(f"{type(s)=}")
        sample_state, _, r, terminated = self.Generator(state, action)
        if terminated:
            cum_reward += r
        else:
            cum_reward += r + self.gamma * self.rollout(sample_state, depth + 1)

        self.replay_buffer.add(initial_state.array, cum_reward)

        return cum_reward

    def Simulate(self, state, h, depth):
        print(f"Simulate\n {state.board} {h=} {depth=}")
        # Check significance of update
        if (self.gamma**depth < self.e or self.gamma == 0) and depth != 0:
            print("Simulation end")
            return 0

        # If leaf node
        if self.tree.is_leaf_node(h):
            action_mask = self.action_mask(state)
            valid_actions = [
                action for i, action in enumerate(self.actions) if action_mask[i]
            ]
            for action in valid_actions:
                print(f"Expanding tree with {action}")
                self.tree.expand_tree_by_one_node(h, action, is_action=True)
            new_value = self.rollout(state, depth)
            self.tree.nodes[h].visit_count += 1
            self.tree.nodes[h].value = new_value
            print("Is leaf node")
            return new_value

        cum_reward = 0
        # Searches best action
        next_action, next_node = self.SearchBest(h, state)
        if next_action is None:
            print("No action!!!")
            return 0
        # Generate next states etc..
        sample_state, sample_observation, reward, terminated = self.Generator(
            state, next_action
        )
        # Get resulting node index
        next_node = self.getObservationNode(next_node, sample_observation)
        # Estimate node Value
        if terminated:
            print(f"Terminated by {next_action}")
            cum_reward += reward
        else:
            print(f"Deeper simulation after {next_action}")
            cum_reward += reward + self.gamma * self.Simulate(
                sample_state, next_node, depth + 1
            )
        # Backtrack
        self.tree.nodes[h].belief_particles.append(state)
        if len(self.tree.nodes[h].belief_particles) > self.no_particles:
            self.tree.nodes[h].belief_particles = self.tree.nodes[h].belief_particles[
                1:
            ]
        self.tree.nodes[h].visit_count += 1
        self.tree.nodes[next_node].visit_count += 1
        self.tree.nodes[next_node].value += (
            cum_reward - self.tree.nodes[next_node].value
        ) / self.tree.nodes[next_node].visit_count
        print(f"cum_reward={cum_reward}")
        return cum_reward

    # FIXFIXFIX
    # Samples from posterior after action and observation
    def PosteriorSample(self, Bh, action, observation):
        if Bh == []:
            state = self.state  # choice(self.state)
        else:
            state = choice(Bh)
        # Sample from transition distribution
        s_next, o_next, _, terminated = self.Generator(state, action)
        if o_next == observation:
            return s_next
        result = self.PosteriorSample(Bh, action, observation)
        return result

    # Updates belief by sampling posterior
    def UpdateBelief(self, action, observation):
        prior = self.tree.nodes[-1].belief_particles.copy()

        self.tree.nodes[-1].belief_particles = []
        for _ in range(self.no_particles):
            self.tree.nodes[-1].belief_particles.append(
                self.PosteriorSample(prior, action, observation)
            )

    def train_value_network(self):
        # Check if replay buffer has enough data
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample a batch of data from the replay buffer
        states, returns = self.replay_buffer.sample()

        states = torch.FloatTensor(states).to(self.device)
        returns = (
            torch.FloatTensor(returns).unsqueeze(1).to(self.device)
        )  # Making returns a column vector

        # Predict the value of each state
        predicted_values = self.value_network(states)

        # Compute the loss
        loss = F.mse_loss(predicted_values, returns)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
