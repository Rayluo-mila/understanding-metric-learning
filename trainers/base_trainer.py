import torch
import os
import time
import psutil
import wandb
import threading
import importlib
import numpy as np
import functools
from collections import deque

from omegaconf import OmegaConf
from pathlib import Path

from utils.logger_wandb import Logger
from agents.distance_functions import metric_func
import utils.rl_utils as rl_utils


def make_agent(obs_shape, action_shape, actor_action_max, cfg, L, max_reward=None, min_reward=None):
    """
    Instantiate the agent using configurations from an additional YAML file.
    """
    # Load the agent-specific configurations from the separate YAML file
    agent_configs = OmegaConf.merge(OmegaConf.load('cfgs/agent_configs.yaml'), OmegaConf.load('cfgs/agent_configs_exp.yaml'))

    # Extract the base agent name (without any suffixes like '_l1' or '_det')
    agent_name = cfg.name
    base_agent_name = agent_name.split('_')[0]

    if base_agent_name not in agent_configs.agents:
        raise NotImplementedError(f'Agent {agent_name} not implemented')

    # Retrieve the specific agent configuration
    agent_cfg = agent_configs.agents[base_agent_name]

    # Dynamically import the agent class
    module_path, class_name = agent_cfg.class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    AgentClass = getattr(module, class_name)

    # Apply agent-specific configurations to cfg.agent
    for key, value in agent_cfg.items():
        if key != 'class_path':
            setattr(cfg, key, value)

    # Check for suffixes and modify configurations accordingly
    if 'det' in agent_name:
        cfg.transition_model_type = 'deterministic'
    elif 'ens2' in agent_name:
        cfg.transition_model_type = 'ensemble_v2'
    elif 'ensd' in agent_name:
        cfg.transition_model_type = 'ensemble_det'
    elif 'ens' in agent_name:
        cfg.transition_model_type = 'ensemble'
    elif 'prob2' in agent_name:
        cfg.transition_model_type = 'probabilistic_v2'
    elif 'prob' in agent_name:
        cfg.transition_model_type = 'probabilistic'

    if 'l1' in agent_name:
        if 'mean' in cfg.bisim_dist: cfg.bisim_dist = 'L1_mean'
        else: cfg.bisim_dist = 'L1'
        cfg.r_dist = 'L1'
    elif 'l2' in agent_name:
        if 'mean' in cfg.bisim_dist: cfg.bisim_dist = 'L2_mean'
        else: cfg.bisim_dist = 'L2'
        cfg.r_dist = 'L2'

    cfg.max_reward = max_reward
    cfg.min_reward = min_reward

    if 'norm' in cfg.encoder_post_processing and (cfg.encoder_type == 'pixel' or cfg.encoder_type == 'mlp'):
        post_processing_str = cfg.encoder_post_processing.replace('_', '')
        cfg.encoder_type += f'_{post_processing_str}ed'
    if 'norm' in cfg.encoder_post_processing_metric and (cfg.encoder_type_metric == 'pixel' or cfg.encoder_type_metric == 'mlp'):
        post_processing_str = cfg.encoder_post_processing_metric.replace('_', '')
        cfg.encoder_type_metric += f'_{post_processing_str}ed'

    print(f'Agent config: {cfg}')

    # Instantiate the agent
    agent = AgentClass(obs_shape, action_shape, actor_action_max, cfg, L)

    return agent


def safe_divide(numerator, denominator, default=1.0):
    """Safely divide two numbers, returning a default value on division by zero."""
    try:
        return numerator / denominator
    except ZeroDivisionError:
        return default


class Timer:
    def __init__(self, enabled_ctx=True):
        self.timings = {}
        self.active_timer = None
        self.enabled_ctx = enabled_ctx
        self.global_start_time = time.time()

    def time(self, name):
        """Context manager for timing a named block."""
        self.active_timer = name
        return self

    def start(self, name):
        """Manually start the timer."""
        if name not in self.timings:
            self.timings[name] = {'start_time': None, 'total_time': 0, 'last_time_period': None}
        
        self.timings[name]['start_time'] = time.time()

    def stop(self, name):
        """Manually stop the timer."""
        start_time = self.timings[name].get('start_time')
        if start_time is None:
            raise RuntimeError(f"Timer '{name}' was not properly started.")
        
        elapsed_time = time.time() - start_time
        self.timings[name]['total_time'] += elapsed_time
        self.timings[name]['last_time_period'] = elapsed_time
        self.timings[name]['start_time'] = None

    def __enter__(self):
        if not self.enabled_ctx: return self
        if self.active_timer is None:
            raise RuntimeError("No timer name set. Use the 'time' method to set a timer name.")
        
        if self.active_timer not in self.timings:
            self.timings[self.active_timer] = {'start_time': None, 'total_time': 0, 'last_time_period': None}
        
        self.timings[self.active_timer]['start_time'] = time.time()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if not self.enabled_ctx: return
        if self.active_timer is None:
            raise RuntimeError("No active timer to stop.")
        
        start_time = self.timings[self.active_timer].get('start_time')
        if start_time is None:
            raise RuntimeError(f"Timer '{self.active_timer}' was not properly started.")

        elapsed_time = time.time() - start_time
        self.timings[self.active_timer]['total_time'] += elapsed_time
        self.timings[self.active_timer]['last_time_period'] = elapsed_time
        self.timings[self.active_timer]['start_time'] = None
        self.active_timer = None
    
    def get_elapsed_time(self, name):
        """Get the total elapsed time for a named timer."""
        return self.timings.get(name, {}).get('total_time', 0)

    def get_last_time_period(self, name):
        """Get the last elapsed time for a named timer within a context."""
        return self.timings.get(name, {}).get('last_time_period', 0)

    def __str__(self):
        """Return a string representation of all timers and their total times."""
        result = ''
        for name, timing in self.timings.items():
            result += f"{name}: {timing['total_time']:.4f} seconds, "
        return result.rstrip().rstrip(',')

    def reset(self, name=None):
        """Reset a named timer."""
        if name is None:
            self.timings = {}
        elif name in self.timings:
            self.timings[name] = {'start_time': None, 'total_time': 0, 'last_time_period': None}

    def get_global_elapsed_time(self):
        """Get the elapsed time since the timer was created."""
        return time.time() - self.global_start_time


class BaseTrainer:
    def __init__(self, cfg):
        self.current_dir = Path.cwd()
        self.work_dir = Path(cfg.work_dir)
        self.cfg = cfg
        self.timer = Timer(cfg.profiling)
        self.start_time = time.time()
        rl_utils.set_seed_everywhere(cfg.seed)

        # Set device
        if cfg.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(cfg.device)

        # Hyperparameter postprocessing
        if cfg.agent.c_T is None:
            assert (cfg.agent.c_R == 1.)
            self.cfg.agent.c_T = cfg.discount
        assert (self.cfg.agent.c_R <= 1. and self.cfg.agent.c_T <= 1.)
        self.cfg.wandb_run_name = None if cfg.wandb_run_name == '' else cfg.wandb_run_name

        # Parallel envs
        if not cfg.use_vectorized_training_env: self.cfg.num_parallel_envs = 1

        # Record task id
        self.cfg.pid = str(os.getpid())
        self.cfg.jobid = str(os.environ["SLURM_JOB_ID"]) if "SLURM_JOB_ID" in os.environ else None
        self.cfg.SLURM_ARRAY_TASK_ID = str(os.environ["SLURM_ARRAY_TASK_ID"]) if "SLURM_ARRAY_TASK_ID" in os.environ else None
        device_name = torch.cuda.get_device_name() if torch.cuda.is_available() else '-'
        self.cfg.GPU_type = device_name

        self.dict_cfg = OmegaConf.to_container(cfg, resolve=True)

        # Create logger
        self.L = Logger(cfg.work_dir, use_wandb=cfg.save_wandb, wandb_run_name=cfg.wandb_run_name,
                        wandb_proj_name=cfg.wandb_proj_name, args=self.dict_cfg, wandb_freq=cfg.wandb_update_freq)

        # Print info
        print(f'Running job with seed={cfg.seed}, agent={cfg.agent.name}')
        print(f'Device: {self.device}, with GPU: {device_name}')
        print(f"Log file: {os.getenv('LOG_FILE')}")
        print(f"Error file: {os.getenv('ERR_FILE')}")

        # Create envs
        self.env = None
        self.eval_env = None

        # Create replay buffer
        self.replay_buffer = None

        # Create agent
        self.agent = None
       
        # For logging train and eval reward diff
        self.train_eps_rew_queue = deque(maxlen=10)

        # For reporting result in the performance table
        assert len(cfg.perf_report_start_step) == len(cfg.perf_report_end_step)
        self.tabular_result_reported = 0
        self.perf_meter = {
            'n': 0,
            'episode_reward': 0,
            'episode_reward_homogeneous': 0,
            'log_div_DF_L2_OOD': 0,
            'log_div_DF_L2_ID': 0,
            'squashed_DF_L2_OOD': 0,
            'squashed_DF_L2_ID': 0,
            }

        self._global_step = 0
        self._global_episode = 0
        self.init(cfg)

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    def init(self, cfg):
        raise NotImplementedError

    def log(self, label, value, log_on_wandb=True):
        if value is None: return
        self.L.log(label, value, self.global_step, log_on_wandb=log_on_wandb)

    def log_multi(self, label_value_dict, log_on_wandb=True):
        for label, value in label_value_dict.items():
            self.log(label, value, log_on_wandb=log_on_wandb)

    def fetch_logger_data(self, label):
        return self.L.get(label)

    def calculate_num_updates(self):
        if not hasattr(self, 'update_counter') and self.global_step >= self.cfg.init_steps:
            self.update_counter = 0.0
            return self.cfg.init_steps
        else:
            expected_updates = self.cfg.num_parallel_envs * self.cfg.replay_ratio
            self.update_counter += expected_updates
            num_updates = int(self.update_counter)
            self.update_counter -= num_updates
            return num_updates

    def eval(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def get_representation(self, obs, encoder):
        with torch.no_grad():
            z_obs = encoder(torch.Tensor(obs).to(self.device))
        return z_obs

    def _compute_sim_score(self, z_anchor_rep, z_samples, dist):
        """
        Compute the similarity score between the anchor and the samples.
        """
        with torch.no_grad():
            mean_dists = dist(x=z_anchor_rep, y=z_samples)
            mean_score = mean_dists.mean().item()
        return mean_dists, mean_score

    def eval_bisimilar_states(self, anchor, pos_samples, neg_samples, dist, encoder, compute_L2=True):
        """
        Compute the similarity score between the anchor and the samples.
        anchor (N(s)): (batch_size_anchor, *obs_shape);  For DMC, (250, 9, 84, 84)
        pos/neg_samples: (batch_size_anchor, batch_size_pos/neg, *obs_shape); For DMC, (250, 8, 9, 84, 84)
        For each anchor, with batch_size_pos pos samples (N'(s)), i.e., batch_size_pos differently noised obses for each anchor.
        For each anchor, with batch_size_neg neg samples (N''(bar_s)), i.e., batch_size_neg obses from different states.
        """
        batch_size_anchor = len(anchor)
        batch_size_pos = len(pos_samples[0])
        batch_size_neg = len(neg_samples[0])

        # Get flat representations
        anchor = np.array(anchor)
        pos_samples = np.array(pos_samples)
        neg_samples = np.array(neg_samples)
        pos_samples_shaped = pos_samples.reshape(-1, *pos_samples.shape[2:])
        neg_samples_shaped = neg_samples.reshape(-1, *neg_samples.shape[2:])
        z_anchor = self.get_representation(anchor, encoder)  # Shape: (batch_size_anchor, representation_dim)
        z_pos = self.get_representation(pos_samples_shaped, encoder)  # Shape: (batch_size_anchor * batch_size_pos, representation_dim)
        z_neg = self.get_representation(neg_samples_shaped, encoder)  # Shape: (batch_size_anchor * batch_size_neg, representation_dim)

        # Repeat z_anchor to match z_pos and z_neg
        z_anchor_pos = z_anchor.repeat_interleave(batch_size_pos, dim=0)  # Shape: (batch_size_anchor * batch_size_pos, representation_dim)
        z_anchor_neg = z_anchor.repeat_interleave(batch_size_neg, dim=0)  # Shape: (batch_size_anchor * batch_size_neg, representation_dim)

        # Compute pairwise distances and mean scores using metric_func
        _, pos_score = self._compute_sim_score(z_anchor_pos, z_pos, dist)  # Shape: (batch_size_anchor * batch_size_pos,)
        _, neg_score = self._compute_sim_score(z_anchor_neg, z_neg, dist)  # Shape: (batch_size_anchor * batch_size_neg,)

        if not compute_L2:
            return pos_score, neg_score, 1, 1

        # metric_func: L2 distance
        dist_L2 = functools.partial(metric_func, cfg=None, override='L2')
        _, pos_score_L2 = self._compute_sim_score(z_anchor_pos, z_pos, dist_L2)
        _, neg_score_L2 = self._compute_sim_score(z_anchor_neg, z_neg, dist_L2)
        return pos_score, neg_score, pos_score_L2, neg_score_L2

    def _record_embeddings(self, obs, action, obses_for_emb, values, embeddings):
        """Record embeddings and values for visualization."""
        obses_for_emb.append(obs)
        with torch.no_grad():
            obs_tensor = torch.Tensor(obs).to(self.device).unsqueeze(0)
            action_tensor = torch.Tensor(action).to(self.device).unsqueeze(0)
            value = min(self.agent.critic(obs_tensor, action_tensor)).item()
            values.append(value)
            embedding = self.agent.critic.encoder(obs_tensor).cpu().detach().numpy()
            embeddings.append(embedding)

    def _record_and_report_performance(self, mean_eps_reward, mean_eps_reward_homo, eval_encoder_result):
        perf_report_times = len(self.cfg.perf_report_start_step)
        if self.tabular_result_reported < perf_report_times:
            if self.global_step >= self.cfg.perf_report_start_step[self.tabular_result_reported] and \
                self.global_step <= self.cfg.perf_report_end_step[self.tabular_result_reported]:
                self.perf_meter['n'] += 1
                self.perf_meter['episode_reward'] += mean_eps_reward
                if eval_encoder_result is not None:
                    self.perf_meter['log_div_DF_L2_ID'] += eval_encoder_result['eval_df_id/log_div_DF_L2']
                    self.perf_meter['squashed_DF_L2_ID'] += eval_encoder_result['eval_df_id/squashed_DF_L2']
                    self.perf_meter['log_div_DF_L2_OOD'] += eval_encoder_result['eval_df_ood/log_div_DF_L2']
                    self.perf_meter['squashed_DF_L2_OOD'] += eval_encoder_result['eval_df_ood/squashed_DF_L2']
                if self.homo_eval_env: 
                    self.perf_meter['episode_reward_homogeneous'] += mean_eps_reward_homo
            if self.global_step >= self.cfg.perf_report_end_step[self.tabular_result_reported]:
                log_step = (self.cfg.perf_report_start_step[self.tabular_result_reported] +
                            self.cfg.perf_report_end_step[self.tabular_result_reported]) // 2
                self.log(f'eval/tab_episode_reward_{log_step}', self.perf_meter['episode_reward'] / self.perf_meter['n'])
                if eval_encoder_result is not None:
                    self.log(f'eval/tab_log_div_DF_L2_ID_{log_step}', self.perf_meter['log_div_DF_L2_ID'] / self.perf_meter['n'])
                    self.log(f'eval/tab_log_div_DF_L2_OOD_{log_step}', self.perf_meter['log_div_DF_L2_OOD'] / self.perf_meter['n'])
                    self.log(f'eval/tab_squashed_DF_L2_ID_{log_step}', self.perf_meter['squashed_DF_L2_ID'] / self.perf_meter['n'])
                    self.log(f'eval/tab_squashed_DF_L2_OOD_{log_step}', self.perf_meter['squashed_DF_L2_OOD'] / self.perf_meter['n'])
                if self.homo_eval_env: 
                    self.log(f'eval_debug/tab_episode_reward_homogeneous_{log_step}',
                             self.perf_meter['episode_reward_homogeneous'] / self.perf_meter['n'])
                self.perf_meter = {key: 0 for key in self.perf_meter.keys()}
                self.tabular_result_reported += 1

    def _log_eval_bism(self, pos_score_sum, neg_score_sum, pos_score_L2_sum, neg_score_L2_sum, n_batch, identifier=''):
        pos_score_mean = pos_score_sum / n_batch
        neg_score_mean = neg_score_sum / n_batch
        pos_score_L2_mean = pos_score_L2_sum / n_batch
        neg_score_L2_mean = neg_score_L2_sum / n_batch
        
        n_div_p = safe_divide(neg_score_sum, pos_score_sum)
        n_div_p_L2 = safe_divide(neg_score_L2_sum, pos_score_L2_sum)
        squashed_DF = safe_divide(neg_score_sum - pos_score_sum, neg_score_sum + pos_score_sum, default=0.0)
        squashed_DF_L2 = safe_divide(neg_score_L2_sum - pos_score_L2_sum, neg_score_L2_sum + pos_score_L2_sum, default=0.0)

        prefix = f'eval_df{identifier}'
        result = {
            f'{prefix}/div_DF': n_div_p,
            f'{prefix}/log_div_DF': np.log(n_div_p).item(),
            f'{prefix}/div_DF_L2': n_div_p_L2,
            f'{prefix}/log_div_DF_L2': np.log(n_div_p_L2).item(),
            f'{prefix}/pos_score': pos_score_mean,
            f'{prefix}/neg_score': neg_score_mean,
            f'{prefix}/pos_score_L2': pos_score_L2_mean,
            f'{prefix}/neg_score_L2': neg_score_L2_mean,
            f'{prefix}/squashed_DF': squashed_DF,
            f'{prefix}/squashed_DF_L2': squashed_DF_L2
        }
        self.log_multi(result)
        return result

    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode', 'env', 'eval_env']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v

    def terminate(self):
        def finish_wandb():
            try:
                wandb.finish()
                print("wandb.finish() completed successfully.")
            except Exception as e:
                print(f"Error in wandb.finish(): {e}")

        print(f"Taking {self.timer.get_global_elapsed_time() / 3600} hrs. Terminating...")
        if self.cfg.save_wandb:
            current_pid = os.getpid()
            finish_thread = threading.Thread(target=finish_wandb)
            finish_thread.start()
            finish_thread.join(timeout=10)

            if finish_thread.is_alive():
                print("wandb.finish() is taking too long, it will show as crashed")
                for proc in psutil.process_iter(["pid", "name", "ppid"]):
                    if "wandb" in proc.info["name"] and proc.info["ppid"] == current_pid:
                        try:
                            proc.terminate()
                            print(f"Terminated: {proc.info['name']}")
                        except psutil.AccessDenied:
                            print(f"Access denied to terminate {proc.info['name']}")
                        except Exception as e:
                            print(f"Failed to terminate {proc.info['name']}: {e}")
            time.sleep(1)
        os._exit(0)