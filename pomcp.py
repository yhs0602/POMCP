import copy
import random

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
        state_transition_function,
        device,
        value_network,
        gamma=0.95,
        ucb_constant=1,
        discount_threshold=0.005,
        timeout=10000,
        num_particles=1200,
        buffer_size=100000,
        batch_size=32,
    ):
        self.gamma = gamma
        if gamma >= 1:
            raise ValueError("gamma should be less than 1.")
        self.state_transition_function = state_transition_function
        self.discount_threshold = discount_threshold
        self.ucb_constant = ucb_constant
        self.timeout = timeout
        self.num_particles = num_particles
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
    def find_best_action(self, node_index, use_ucb=True):
        max_value = None
        result = None
        resulta = None
        if use_ucb:
            children = self.tree.nodes[node_index].children
            # UCB for each child node
            for action, child in children.items():
                # if node is unvisited return it
                if self.tree.nodes[child].visit_count == 0:
                    return action, child
                ucb = UCB(
                    self.tree.nodes[node_index].visit_count,
                    self.tree.nodes[child].visit_count,
                    self.tree.nodes[child].value,
                    self.ucb_constant,
                )

                # Max is kept
                if max_value is None or max_value < ucb:
                    max_value = ucb
                    result = child
                    resulta = action
            # return action-child_id values
            return resulta, result
        else:
            children = self.tree.nodes[node_index].children
            # pick optimal value node for termination
            for action, child in children.items():
                node_value = self.tree.nodes[child].value
                # keep max
                if max_value is None or max_value < node_value:
                    max_value = node_value
                    result = child
                    resulta = action
            return resulta, result

    # Search module
    def search(self):
        current_belief_particles = self.tree.nodes[-1].belief_particles.copy()
        print(f"Search {current_belief_particles=}")
        # Repeat Simulations until timeout
        for _ in range(self.timeout):
            if not current_belief_particles:
                state = self.state  # choice(self.states)
            else:
                state = choice(list(current_belief_particles))
            self.perform_simulation(copy.deepcopy(state), -1, 0)
        # Get best action
        action, _ = self.find_best_action(-1, use_ucb=False)
        return action

    # Check if a given observation node has been visited
    def retrieve_or_create_observation_node(self, node_index, sample_observation):
        if sample_observation not in list(self.tree.nodes[node_index].children.keys()):
            # If not create the node
            self.tree.expand_tree_by_one_node(node_index, sample_observation)
        # Get the nodes index
        next_node = self.tree.nodes[node_index].children[sample_observation]
        return next_node

    def rollout(self, state: FowState, depth):
        # Check significance of update
        if (
            self.gamma**depth < self.discount_threshold or self.gamma == 0
        ) and depth != 0:
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
        action = self.choose_random_valid_action(state)
        # print(f"{valid_actions=} {action=}")
        # action = choice(self.actions)

        # Generate states and observations
        # print(f"{type(s)=}")
        sample_state, _, r, terminated = self.state_transition_function(state, action)
        if terminated:
            cum_reward += r
        else:
            cum_reward += r + self.gamma * self.rollout(sample_state, depth + 1)

        self.replay_buffer.add(initial_state.array, cum_reward)

        return cum_reward

    def choose_random_valid_action(self, state):
        action_mask = self.action_mask(state)
        valid_actions = np.where(action_mask == 1)[0]
        action_idx = np.random.choice(valid_actions)
        action = self.actions[action_idx]
        return action

    def perform_simulation(self, state, node_index, depth):
        # print(f"Performing simulation from {node_index=} {depth=}")
        # print(f"Simulate\n {state.board} {node_index=} {depth=}")
        # Check significance of update
        if (
            self.gamma**depth < self.discount_threshold or self.gamma == 0
        ) and depth != 0:
            # print("Simulation end")
            return 0

        # If leaf node, need to expand the tree
        if self.tree.is_leaf_node(node_index):
            valid_actions = self.get_valid_actions(state)
            # print(f"Leaf node, expanding tree {node_index} with")
            board = state.board
            for action in valid_actions:
                # print(f"{action}={str(action_index_to_move(board, action))}")
                self.tree.expand_tree_by_one_node(node_index, action, is_action=True)
            new_value = self.rollout(state, depth)
            self.tree.nodes[node_index].visit_count += 1
            self.tree.nodes[node_index].value = new_value
            return new_value

        cum_reward = 0
        # Searches best action
        next_action, next_node = self.find_best_action(node_index, state)
        if next_action is None:
            print("No action!!!")
            return 0
        # Generate next states etc..
        (
            sample_state,
            sample_observation,
            reward,
            terminated,
        ) = self.state_transition_function(state, next_action)
        # Get resulting node index
        next_node = self.retrieve_or_create_observation_node(
            next_node, sample_observation
        )
        # Estimate node Value
        if terminated:
            # print(f"Terminated by {next_action}")
            cum_reward += reward
        else:
            # print(f"Deeper simulation after {next_action}")
            cum_reward += reward + self.gamma * self.perform_simulation(
                sample_state, next_node, depth + 1
            )
        # Backtrack
        self.tree.nodes[node_index].belief_particles.add(state)
        if len(self.tree.nodes[node_index].belief_particles) > self.num_particles:
            self.tree.nodes[node_index].belief_particles = set(
                random.sample(
                    list(self.tree.nodes[node_index].belief_particles),
                    self.num_particles,
                )
            )
        self.tree.nodes[node_index].visit_count += 1
        self.tree.nodes[next_node].visit_count += 1
        self.tree.nodes[next_node].value += (
            cum_reward - self.tree.nodes[next_node].value
        ) / self.tree.nodes[next_node].visit_count
        # print(f"cum_reward={cum_reward}")
        return cum_reward

    def get_valid_actions(self, state):
        action_mask = self.action_mask(state)
        valid_actions = [
            action for i, action in enumerate(self.actions) if action_mask[i]
        ]
        return valid_actions

    # FIXFIXFIX
    # Samples from posterior after action and observation
    def posterior_sampling(self, current_belief_particles, action, observation):
        max_attempts = 100
        for _ in range(max_attempts):
            if not current_belief_particles:
                state = self.state
            else:
                state = choice(list(current_belief_particles))
            s_next, o_next, _, terminated = self.state_transition_function(
                state, action
            )
            if o_next == observation:
                return s_next
        return None  # or handle this case as needed

    # Updates belief by sampling posterior
    def update_belief(self, action, observation):
        prior = self.tree.nodes[-1].belief_particles.copy()

        self.tree.nodes[-1].belief_particles = set()
        for _ in range(self.num_particles):
            posterior = self.posterior_sampling(prior, action, observation)
            if posterior is not None:
                self.tree.nodes[-1].belief_particles.add(posterior)

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
