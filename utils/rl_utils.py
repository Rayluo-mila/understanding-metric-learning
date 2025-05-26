import torch
import numpy as np
import os
import random


def count_parameters(model, recurse=True):
    if model is None:
        return 0
    return sum(p.numel() for p in model.parameters(recurse=recurse) if p.requires_grad)


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def set_seed_everywhere(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def module_hash(module):
    result = 0
    for tensor in module.state_dict().values():
        result += tensor.sum().item()
    return result


def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2**bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2 ** (8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


class ReplayBuffer(object):
    """Buffer to store environment transitions."""

    def __init__(self, obs_shape, action_shape, capacity, batch_size, device, nprocs=1):
        print(
            f"Building replay buffer with capacity={capacity}, B={batch_size}, obs={obs_shape}"
        )
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.nprocs = nprocs

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.k_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.curr_rewards = np.empty((capacity, 1), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        # compute memory needed
        obs_elements = np.prod(
            (capacity, *obs_shape)
        )  # Total number of elements in the array
        action_elements = np.prod((capacity, *action_shape))
        reward_elements = np.prod((capacity, 1))
        obs_memory = obs_elements * self.obses.itemsize / (1024**3)
        action_memory = action_elements * self.actions.itemsize / (1024**3)
        reward_memory = reward_elements * self.rewards.itemsize / (1024**3)
        print(
            f"Mem: obses need {obs_memory:.2f} GB, action need {action_memory:.2f} GB, reward need {reward_memory:.2f} GB"
        )

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs, action, curr_reward, reward, next_obs, done):
        if self.nprocs == 1:
            self.add_single(obs, action, curr_reward, reward, next_obs, done)
        else:
            self.add_multi(obs, action, curr_reward, reward, next_obs, done)

    def add_single(self, obs, action, curr_reward, reward, next_obs, done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.curr_rewards[self.idx], curr_reward)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def add_multi(self, obs, action, curr_reward, reward, next_obs, done):
        # Calculate the number of elements that can be added without overflow
        available_space = self.capacity - self.idx
        num_to_add = min(self.nprocs, available_space)

        # Add elements that fit within the current capacity
        np.copyto(self.obses[self.idx : self.idx + num_to_add], obs[:num_to_add])
        np.copyto(self.actions[self.idx : self.idx + num_to_add], action[:num_to_add])
        np.copyto(
            self.curr_rewards[self.idx : self.idx + num_to_add],
            np.expand_dims(curr_reward[:num_to_add], axis=-1),
        )
        np.copyto(
            self.rewards[self.idx : self.idx + num_to_add],
            np.expand_dims(reward[:num_to_add], axis=-1),
        )
        np.copyto(
            self.next_obses[self.idx : self.idx + num_to_add], next_obs[:num_to_add]
        )
        np.copyto(
            self.not_dones[self.idx : self.idx + num_to_add],
            np.expand_dims([not _ for _ in done[:num_to_add]], axis=-1),
        )

        # Update index and handle overflow
        self.idx = (self.idx + num_to_add) % self.capacity
        self.full = self.full or self.idx == 0

        # Handle overflow by wrapping around and adding remaining elements
        if num_to_add < self.nprocs:
            remaining = self.nprocs - num_to_add
            np.copyto(self.obses[:remaining], obs[num_to_add : num_to_add + remaining])
            np.copyto(
                self.actions[:remaining], action[num_to_add : num_to_add + remaining]
            )
            np.copyto(
                self.curr_rewards[:remaining],
                np.expand_dims(
                    curr_reward[num_to_add : num_to_add + remaining], axis=-1
                ),
            )
            np.copyto(
                self.rewards[:remaining],
                np.expand_dims(reward[num_to_add : num_to_add + remaining], axis=-1),
            )
            np.copyto(
                self.next_obses[:remaining],
                next_obs[num_to_add : num_to_add + remaining],
            )
            np.copyto(
                self.not_dones[:remaining],
                np.expand_dims(
                    [not _ for _ in done[num_to_add : num_to_add + remaining]], axis=-1
                ),
            )

            self.idx = remaining % self.capacity

    def sample(self, k=False):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        curr_rewards = torch.as_tensor(self.curr_rewards[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        if k:
            return (
                obses,
                actions,
                rewards,
                next_obses,
                not_dones,
                torch.as_tensor(self.k_obses[idxs], device=self.device),
            )
        return obses, actions, curr_rewards, rewards, next_obses, not_dones

    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, "%d_%d.pt" % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save : self.idx],
            self.next_obses[self.last_save : self.idx],
            self.actions[self.last_save : self.idx],
            self.rewards[self.last_save : self.idx],
            self.curr_rewards[self.last_save : self.idx],
            self.not_dones[self.last_save : self.idx],
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split("_")[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split(".")[0].split("_")]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.curr_rewards[start:end] = payload[4]
            self.not_dones[start:end] = payload[5]
            self.idx = end


class MemSaveReplayBuffer(object):
    """
    Buffer to store environment transitions w/o next_obses.
    Pay attention to handle the next observation when done.
    """

    def __init__(self, obs_shape, action_shape, capacity, batch_size, device, nprocs=1):
        print(
            f"Building replay buffer with capacity={capacity}, B={batch_size}, obs={obs_shape}, action={action_shape}"
        )
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.nprocs = nprocs

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity + 1, *obs_shape), dtype=obs_dtype)
        self.k_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.curr_rewards = np.empty((capacity, 1), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        # buffer for next_obs when done=True
        self.done_next_obses = {}

        # compute memory needed
        obs_elements = np.prod(
            (capacity + 1, *obs_shape)
        )  # Total number of elements in the array
        action_elements = np.prod((capacity, *action_shape))
        reward_elements = np.prod((capacity, 1))
        obs_memory = obs_elements * self.obses.itemsize / (1024**3)
        action_memory = action_elements * self.actions.itemsize / (1024**3)
        reward_memory = reward_elements * self.rewards.itemsize / (1024**3)
        print(
            f"Mem: obses need {obs_memory:.2f} GB, action need {action_memory:.2f} GB, reward need {reward_memory:.2f} GB"
        )

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs, action, curr_reward, reward, next_obs, done):
        if self.nprocs == 1:
            self.add_single(obs, action, curr_reward, reward, next_obs, done)
        else:
            self.add_multi(obs, action, curr_reward, reward, next_obs, done)

    def add_single(self, obs, action, curr_reward, reward, next_obs, done):
        if self.idx != 0 and self.not_dones[self.idx - 1]:
            pass
        else:
            np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.curr_rewards[self.idx], curr_reward)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.not_dones[self.idx], float(not done))

        next_idx = (self.idx + 1) % self.capacity
        if done:
            self.done_next_obses[self.idx] = next_obs
        else:
            np.copyto(self.obses[next_idx], next_obs)

        self.idx = next_idx
        self.full = self.full or self.idx == 0

    def add_multi(self, obs, action, curr_reward, reward, next_obs, done):
        # Calculate the number of elements that can be added without overflow
        available_space = self.capacity - self.idx
        num_to_add = min(self.nprocs, available_space)

        # Add elements that fit within the current capacity
        np.copyto(self.obses[self.idx : self.idx + num_to_add], obs[:num_to_add])
        np.copyto(self.actions[self.idx : self.idx + num_to_add], action[:num_to_add])
        np.copyto(
            self.curr_rewards[self.idx : self.idx + num_to_add],
            np.expand_dims(curr_reward[:num_to_add], axis=-1),
        )
        np.copyto(
            self.rewards[self.idx : self.idx + num_to_add],
            np.expand_dims(reward[:num_to_add], axis=-1),
        )
        np.copyto(
            self.not_dones[self.idx : self.idx + num_to_add],
            np.expand_dims([float(not d) for d in done[:num_to_add]], axis=-1),
        )

        for i in range(num_to_add):
            if done[i]:
                self.done_next_obses[(self.idx + i) % self.capacity] = next_obs[i]
            else:
                np.copyto(
                    self.obses[(self.idx + i + self.nprocs) % self.capacity],
                    next_obs[i],
                )

        # Update index and handle overflow
        self.idx = (self.idx + num_to_add) % self.capacity
        self.full = self.full or self.idx == 0

        # Handle overflow by wrapping around and adding remaining elements
        if num_to_add < self.nprocs:
            remaining = self.nprocs - num_to_add
            np.copyto(self.obses[:remaining], obs[num_to_add : num_to_add + remaining])
            np.copyto(
                self.actions[:remaining], action[num_to_add : num_to_add + remaining]
            )
            np.copyto(
                self.curr_rewards[:remaining],
                np.expand_dims(
                    curr_reward[num_to_add : num_to_add + remaining], axis=-1
                ),
            )
            np.copyto(
                self.rewards[:remaining],
                np.expand_dims(reward[num_to_add : num_to_add + remaining], axis=-1),
            )
            np.copyto(
                self.not_dones[:remaining],
                np.expand_dims(
                    [float(not d) for d in done[num_to_add : num_to_add + remaining]],
                    axis=-1,
                ),
            )

            for i in range(remaining):
                if done[num_to_add + i]:
                    self.done_next_obses[i] = next_obs[num_to_add + i]
                else:
                    np.copyto(
                        self.obses[(i + self.nprocs) % self.capacity],
                        next_obs[num_to_add + i],
                    )

            self.idx = remaining % self.capacity

    def sample(self, k=False):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )
        next_idxs = (idxs + self.nprocs) % self.capacity

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        curr_rewards = torch.as_tensor(self.curr_rewards[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        # Preallocate the next_obses tensor using the next observations calculated in bulk
        next_obses = torch.as_tensor(
            self.obses[next_idxs], device=self.device, dtype=torch.float32
        )

        # Create a mask where done is True (i.e., where we need to replace with done_next_obses)
        done_mask = torch.tensor(
            [idx in self.done_next_obses for idx in idxs], device=self.device
        )

        # Collect the indices where done is True
        done_indices = torch.where(done_mask)[0]

        # Gather the corresponding done_next_obses and replace them in the preallocated tensor
        if done_indices.numel() > 0:
            done_next_obs_list = np.array(
                [self.done_next_obses[idxs[i]] for i in done_indices.cpu().numpy()]
            )
            done_next_obses = torch.as_tensor(
                done_next_obs_list, device=self.device, dtype=torch.float32
            )
            next_obses[done_indices] = done_next_obses

        if k:  # need further inspection
            return (
                obses,
                actions,
                rewards,
                next_obses,
                not_dones,
                torch.as_tensor(self.k_obses[idxs], device=self.device),
            )
        return obses, actions, curr_rewards, rewards, next_obses, not_dones

    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, "%d_%d.pt" % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save : self.idx],
            self.actions[self.last_save : self.idx],
            self.rewards[self.last_save : self.idx],
            self.curr_rewards[self.last_save : self.idx],
            self.not_dones[self.last_save : self.idx],
            self.done_next_obses,
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def save_obs(self, save_dir, name):
        assert self.full
        path = os.path.join(save_dir, f"obs_{name}.pt")
        torch.save(self.obses, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split("_")[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split(".")[0].split("_")]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.actions[start:end] = payload[1]
            self.rewards[start:end] = payload[2]
            self.curr_rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            self.done_next_obses = payload[5]
            self.idx = end


class ObsMemory(object):
    """
    A lightweight buffer to only store observations.
    """

    def __init__(
        self, obs_shape, capacity, batch_size, device, nprocs=1, clean_obs_shape=None
    ):
        print(
            f"Building observation memory with capacity={capacity}, B={batch_size}, obs={obs_shape}"
        )
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.nprocs = nprocs
        self.obs_shape = obs_shape
        self.clean_obs_shape = clean_obs_shape

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        self.obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8
        self.obses = np.empty((capacity, *obs_shape), dtype=self.obs_dtype)
        if clean_obs_shape is not None:
            self.clean_obses = np.empty(
                (capacity, *clean_obs_shape), dtype=self.obs_dtype
            )

        # compute memory needed
        obs_elements = np.prod(
            (capacity, *obs_shape)
        )  # Total number of elements in the array
        obs_memory = obs_elements * self.obses.itemsize / (1024**3)
        print(f"ObsMem: need {obs_memory:.2f} GB")

        self.idx = 0
        self.batch_number = 0
        self.full = False

    def add(self, obs, clean_obs=None):
        if self.nprocs == 1:
            self.add_single(obs, clean_obs)
        else:
            self.add_multi(obs, clean_obs)

    def add_single(self, obs, clean_obs=None):
        np.copyto(self.obses[self.idx], obs)
        if clean_obs is not None:
            np.copyto(self.clean_obses[self.idx], clean_obs)
        next_idx = (self.idx + 1) % self.capacity
        self.idx = next_idx
        self.full = self.full or self.idx == 0

    def add_multi(self, obs, clean_obs=None):
        # Calculate the number of elements that can be added without overflow
        available_space = self.capacity - self.idx
        num_to_add = min(self.nprocs, available_space)

        # Add elements that fit within the current capacity
        np.copyto(self.obses[self.idx : self.idx + num_to_add], obs[:num_to_add])
        if clean_obs is not None:
            np.copyto(
                self.clean_obses[self.idx : self.idx + num_to_add],
                clean_obs[:num_to_add],
            )

        # Update index and handle overflow
        self.idx = (self.idx + num_to_add) % self.capacity
        self.full = self.full or self.idx == 0

        # Handle overflow by wrapping around and adding remaining elements
        if num_to_add < self.nprocs:
            remaining = self.nprocs - num_to_add
            np.copyto(self.obses[:remaining], obs[num_to_add : num_to_add + remaining])
            if clean_obs is not None:
                np.copyto(
                    self.clean_obses[:remaining],
                    clean_obs[num_to_add : num_to_add + remaining],
                )
            self.idx = remaining % self.capacity

    def clear(self):
        self.idx = 0
        self.full = False
        self.obses = np.empty((self.capacity, *self.obs_shape), dtype=self.obs_dtype)
        if self.clean_obs_shape is not None:
            self.clean_obses = np.empty(
                (self.capacity, *self.clean_obs_shape), dtype=self.obs_dtype
            )

    def sample(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=batch_size
        )
        obses = np.array(self.obses[idxs], dtype=np.float32)
        return obses

    def sample_clean_obs(self, batch_size=None):
        if self.clean_obs_shape is None:
            return None
        batch_size = self.batch_size if batch_size is None else batch_size
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=batch_size
        )
        clean_obses = np.array(self.clean_obses[idxs], dtype=np.float32)
        return clean_obses

    def save_all(self, save_dir, name):
        assert self.full
        path = os.path.join(save_dir, f"obsmem_{name}_{self.batch_number}.pt")
        self.batch_number += 1
        torch.save(self.obses, path)


class Struct:
    def __init__(self, **entries):
        rec_entries = {}
        for k, v in entries.items():
            if isinstance(v, dict):
                rv = Struct(**v)
            elif isinstance(v, list):
                rv = []
                for item in v:
                    if isinstance(item, dict):
                        rv.append(Struct(**item))
                    else:
                        rv.append(item)
            else:
                rv = v
            rec_entries[k] = rv
        self.__dict__.update(rec_entries)

    def __str_helper(self, depth):
        lines = []
        for k, v in self.__dict__.items():
            if isinstance(v, Struct):
                v_str = v.__str_helper(depth + 1)
                lines.append("%s:\n%s" % (k, v_str))
            else:
                lines.append("%s: %r" % (k, v))
        indented_lines = ["    " * depth + l for l in lines]
        return "\n".join(indented_lines)

    def __str__(self):
        return "struct {\n%s\n}" % self.__str_helper(1)

    def __repr__(self):
        return "Struct(%r)" % self.__dict__
