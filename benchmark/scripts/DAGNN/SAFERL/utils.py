import numpy as np
import torch
def replace_nan_with_column_mean(array):
    # Create masks for NaN, -NaN, inf, and -inf
    nan_mask = np.isnan(array)
    inf_mask = np.isinf(array)

    # Compute the mean of each column, ignoring NaN, -NaN, inf, and -inf
    # Replace inf and -inf with NaN for mean calculation
    clean_array = np.where(inf_mask, np.nan, array)
    column_means = np.nanmean(clean_array, axis=0)

    # Replace NaNs and infs with the column mean
    array[nan_mask] = np.take(column_means, np.where(nan_mask)[1])
    array[inf_mask] = np.take(column_means, np.where(inf_mask)[1])

    return array

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(40000)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        temp1=np.where(np.isnan(self.state))
        temp2=np.where(np.isnan(self.next_state))
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.costs[ind]).to(self.device)
        )

    def convert_D4RL(self, dataset):
        self.state = dataset['observations']

        self.action = dataset['actions']
        self.next_state = dataset['next_observations']
        self.reward = dataset['rewards'].reshape(-1, 1)
        self.not_done = 1. - dataset['terminals'].reshape(-1, 1)



    def convert_csv(self, path="offline/"):
        # Read CSV files into numpy arraysq

        state = np.loadtxt(path+"observations.csv", delimiter=',')
        self.state = replace_nan_with_column_mean(state)[::2]
        action = np.loadtxt(path+"actions.csv", delimiter=',',dtype=int)/8

        # Transform the original array to a (144000, 9) array
        self.action = action.reshape(-1,1)[1::2]
        next_state = np.loadtxt(path+"next_observations.csv", delimiter=',')
        self.next_state = replace_nan_with_column_mean(next_state)[::2]
        self.reward = -np.loadtxt(path+"rewards.csv", delimiter=',').reshape(-1, 1)[1::2]
        self.not_done = 1. - np.loadtxt(path+"terminated.csv", delimiter=',').reshape(-1, 1)[1::2]
        self.costs = -np.loadtxt(path+"constraints.csv", delimiter=',')[1::2]

        # Determine the size of the dataset

        self.size = self.state.shape[0]
        print(self.state.shape)
        print(self.action.shape)
        print(self.next_state.shape)
        print(self.reward.shape)
        print(self.not_done.shape)
        self.size = self.state.shape[0]


    def normalize_states(self, eps=1e-3, mean=None, std=None):
        if mean is None and std is None:
            mean = self.state.mean(0, keepdims=True)
            std = self.state.std(0, keepdims=True) + eps
        self.state = (self.state - mean) / std
        self.next_state = (self.next_state - mean) / std
        return mean, std
