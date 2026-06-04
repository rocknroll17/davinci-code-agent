"""
Vectorized Environment for Da Vinci Code.

Manages multiple game environments in parallel for efficient training.
Includes both thread-based (VectorDaVinciEnv) and multiprocessing-based
(SubprocVecEnv) implementations.
"""

import os
import numpy as np
import multiprocessing as mp
from typing import List, Tuple, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from src.env import DaVinciCodeEnv



# ============================================================
# Worker process function for SubprocVecEnv
# ============================================================
def _worker_loop(pipe, parent_pipe, n_local_envs, seed_base, reward_config=None):
    """
    Main loop for a worker process.
    Each worker manages n_local_envs environments.
    """
    parent_pipe.close()
    
    envs = [
        DaVinciCodeEnv(seed=seed_base + i if seed_base is not None else None, reward_config=reward_config)
        for i in range(n_local_envs)
    ]
    
    while True:
        try:
            cmd, data = pipe.recv()
        except (EOFError, BrokenPipeError):
            break
        
        if cmd == 'step':
            actions = data
            obs_list = []
            rewards = np.zeros(n_local_envs, dtype=np.float32)
            terminated = np.zeros(n_local_envs, dtype=bool)
            truncated_arr = np.zeros(n_local_envs, dtype=bool)
            info_list = []
            result_list = []
            
            for idx, (env, action) in enumerate(zip(envs, actions)):
                o, _, r, te, tr, info, result = env.step(action)
                # Auto-reset: include winner + reset obs for done envs
                if te or tr:
                    info['_winner'] = env._winner
                    reset_obs, reset_info = env.reset()
                    info['_reset_obs'] = reset_obs
                    info['_reset_info'] = reset_info
                obs_list.append(o)
                rewards[idx] = r
                terminated[idx] = te
                truncated_arr[idx] = tr
                info_list.append(info)
                result_list.append(result)
            
            pipe.send((obs_list, rewards, terminated, truncated_arr, info_list, result_list))
        
        elif cmd == 'reset':
            results = [env.reset() for env in envs]
            obs_list = [r[0] for r in results]
            info_list = [r[1] for r in results]
            pipe.send((obs_list, info_list))
        
        elif cmd == 'get_masks':
            masks = [env.get_action_mask() for env in envs]
            pipe.send(masks)
        
        elif cmd == 'close':
            for env in envs:
                env.close()
            break
    
    pipe.close()


class SubprocVecEnv:
    """
    Multiprocessing-based vectorized environment.
    
    Distributes n_envs across multiple worker processes for true
    CPU parallelism (bypasses GIL). Each worker manages a chunk of envs.
    
    Key advantages over VectorDaVinciEnv (thread-based):
    - True parallelism for CPU-bound env.step()
    - O(n_envs / n_workers) per step instead of O(n_envs)
    - Auto-reset: done envs reset inside workers (no extra round-trip)
    """
    
    def __init__(self, n_envs: int = 2000, seed: Optional[int] = None, n_workers: Optional[int] = None,
                 reward_config=None) -> None:
        self.reward_config = reward_config
        self.n_envs = n_envs
        self.n_workers = n_workers or min(os.cpu_count() * 2 or 4, max(1, n_envs // 30))
        self.n_workers = max(1, min(self.n_workers, n_envs))
        
        # Distribute envs across workers as evenly as possible
        base_count = n_envs // self.n_workers
        remainder = n_envs % self.n_workers
        self.envs_per_worker = []
        for w in range(self.n_workers):
            self.envs_per_worker.append(base_count + (1 if w < remainder else 0))
        
        # Cumulative offsets: worker_offsets[w] = first global index of worker w
        self.worker_offsets = [0]
        for count in self.envs_per_worker:
            self.worker_offsets.append(self.worker_offsets[-1] + count)
        
        # Spawn worker processes
        self.pipes: List[mp.connection.Connection] = []
        self.workers: List[mp.Process] = []
        
        ctx = mp.get_context('fork')  # fork is faster than spawn on Linux
        
        seed_offset = 0
        for w in range(self.n_workers):
            parent_conn, child_conn = ctx.Pipe()
            n_local = self.envs_per_worker[w]
            worker_seed = (seed + seed_offset) if seed is not None else None
            
            worker = ctx.Process(
                target=_worker_loop,
                args=(child_conn, parent_conn, n_local, worker_seed, reward_config),
                daemon=True
            )
            worker.start()
            child_conn.close()
            
            self.pipes.append(parent_conn)
            self.workers.append(worker)
            seed_offset += n_local
        
        # Local env for visualization only (not used in training)
        self._viz_env = DaVinciCodeEnv(seed=seed if seed is not None else None, reward_config=reward_config)
        
        print(f"SubprocVecEnv: {n_envs} envs across {self.n_workers} workers "
              f"({self.envs_per_worker[0]}-{self.envs_per_worker[-1]} envs/worker)")
    
    def reset(self) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]]]:
        """Reset all environments (parallel across workers)."""
        for pipe in self.pipes:
            pipe.send(('reset', None))
        
        all_obs = []
        all_infos = []
        for pipe in self.pipes:
            obs_list, info_list = pipe.recv()
            all_obs.extend(obs_list)
            all_infos.extend(info_list)
        
        return self._batch_obs(all_obs), all_infos
    
    def step(
        self,
        actions: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, List[Dict], List]:
        """
        Step all environments in parallel.
        Done envs are auto-reset inside workers.
        Reset obs available in info['_reset_obs'] for done envs.
        """
        # Send actions to each worker
        for w in range(self.n_workers):
            start = self.worker_offsets[w]
            end = self.worker_offsets[w + 1]
            self.pipes[w].send(('step', actions[start:end]))
        
        # Collect results
        all_obs = []
        all_rewards = []
        all_terminated = []
        all_truncated = []
        all_infos = []
        all_results = []
        
        for pipe in self.pipes:
            obs_list, rewards, terminated, truncated_arr, info_list, result_list = pipe.recv()
            all_obs.extend(obs_list)
            all_rewards.append(rewards)
            all_terminated.append(terminated)
            all_truncated.append(truncated_arr)
            all_infos.extend(info_list)
            all_results.extend(result_list)
        
        return (
            self._batch_obs(all_obs),
            np.concatenate(all_rewards),
            np.concatenate(all_terminated),
            np.concatenate(all_truncated),
            all_infos,
            all_results
        )
    
    def get_action_masks(self) -> Dict[str, np.ndarray]:
        """Get action masks for all environments (parallel)."""
        for pipe in self.pipes:
            pipe.send(('get_masks', None))
        
        all_masks = []
        for pipe in self.pipes:
            masks = pipe.recv()
            all_masks.extend(masks)
        
        return {
            key: np.stack([m[key] for m in all_masks])
            for key in all_masks[0].keys()
        }
    
    def reset_single(self, env_idx: int) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Not needed with auto-reset, but kept for API compatibility."""
        raise NotImplementedError(
            "SubprocVecEnv uses auto-reset. Access info['_reset_obs'] for done envs."
        )
    
    def get_viz_env(self) -> DaVinciCodeEnv:
        """Get local env for visualization (not part of training)."""
        return self._viz_env
    
    def _batch_obs(self, obs_list: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        return {
            key: np.stack([obs[key] for obs in obs_list])
            for key in obs_list[0].keys()
        }
    
    def close(self) -> None:
        """Shut down all worker processes."""
        for pipe in self.pipes:
            try:
                pipe.send(('close', None))
            except (BrokenPipeError, OSError):
                pass
        for worker in self.workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()


# ============================================================
# Original thread-based implementation (kept for compatibility)
# ============================================================

def _step_single_env(args):
    """Helper function for parallel stepping."""
    env, action = args
    return env.step(action)


class VectorDaVinciEnv:
    """
    Thread-based vectorized wrapper for multiple DaVinciCodeEnv instances.
    
    Good for small n_envs (<500) or debugging. For large n_envs,
    use SubprocVecEnv for true CPU parallelism.
    """
    
    def __init__(self, n_envs: int = 8, seed: Optional[int] = None, use_threads: bool = True,
                 reward_config=None) -> None:
        self.n_envs = n_envs
        self.envs = [
            DaVinciCodeEnv(seed=seed + i if seed else None, reward_config=reward_config)
            for i in range(n_envs)
        ]
        
        self.use_threads = use_threads and n_envs > 1
        if self.use_threads:
            self.executor = ThreadPoolExecutor(max_workers=min(n_envs, 8))
        
        self.viz_env_idx = 0
    
    def reset(self) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]]]:
        obs_list = []
        info_list = []
        for env in self.envs:
            obs, info = env.reset()
            obs_list.append(obs)
            info_list.append(info)
        return self._batch_obs(obs_list), info_list
    
    def reset_single(self, env_idx: int) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        return self.envs[env_idx].reset()
    
    def step(
        self,
        actions: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, List[Dict], List]:
        next_obs_list = []
        rewards = np.zeros(self.n_envs, dtype=np.float32)
        terminated = np.zeros(self.n_envs, dtype=bool)
        truncated = np.zeros(self.n_envs, dtype=bool)
        info_list = []
        result_list = []
        
        if self.use_threads:
            args = list(zip(self.envs, actions))
            results = list(self.executor.map(_step_single_env, args))
            for i, (next_obs, _, reward, term, trunc, info, result) in enumerate(results):
                next_obs_list.append(next_obs)
                rewards[i] = reward
                terminated[i] = term
                truncated[i] = trunc
                info_list.append(info)
                result_list.append(result)
        else:
            for i, (env, action) in enumerate(zip(self.envs, actions)):
                next_obs, _, reward, term, trunc, info, result = env.step(action)
                next_obs_list.append(next_obs)
                rewards[i] = reward
                terminated[i] = term
                truncated[i] = trunc
                info_list.append(info)
                result_list.append(result)
        
        return (
            self._batch_obs(next_obs_list),
            rewards, terminated, truncated,
            info_list, result_list
        )
    
    def get_action_masks(self) -> Dict[str, np.ndarray]:
        masks = [env.get_action_mask() for env in self.envs]
        return {
            key: np.stack([m[key] for m in masks])
            for key in masks[0].keys()
        }
    
    def get_current_players(self) -> np.ndarray:
        return np.array([env._current_player for env in self.envs])
    
    def get_viz_env(self) -> DaVinciCodeEnv:
        return self.envs[self.viz_env_idx]
    
    def _batch_obs(self, obs_list: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        return {
            key: np.stack([obs[key] for obs in obs_list])
            for key in obs_list[0].keys()
        }
    
    def close(self) -> None:
        if self.use_threads:
            self.executor.shutdown(wait=False)
        for env in self.envs:
            env.close()
