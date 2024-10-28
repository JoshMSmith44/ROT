import warnings

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
from pathlib import Path

import hydra
import numpy as np
import torch
from dm_env import specs

import utils
from logger import Logger
from video import TrainVideoRecorder, VideoRecorder
import pickle
import cv2

torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore', category=DeprecationWarning)


def make_agent(obs_spec, action_spec, cfg):
	cfg.obs_shape = obs_spec['pixels'].shape
	cfg.action_shape = action_spec.shape
	return hydra.utils.instantiate(cfg)


class Workspace:
	def __init__(self, cfg):
		self.work_dir = Path.cwd()
		print(f'workspace: {self.work_dir}')

		self.cfg = cfg
		utils.set_seed_everywhere(cfg.seed)
		self.device = torch.device(cfg.device)
		self.setup()

		self.agent = make_agent(self.train_env.observation_spec(),
								self.train_env.action_spec(), cfg.agent)
		self.timer = utils.Timer()
		self._global_step = 0
		self._global_episode = 0

	def setup(self):
		# create envs
		self.train_env = hydra.utils.call(self.cfg.suite.task_make_fn)
		self.eval_env = hydra.utils.call(self.cfg.suite.task_make_fn)

		self.video_recorder = VideoRecorder(
			self.work_dir if self.cfg.save_video else None)
		self.train_video_recorder = TrainVideoRecorder(
			self.work_dir if self.cfg.save_train_video else None)
		

	@property
	def global_step(self):
		return self._global_step

	@property
	def global_episode(self):
		return self._global_episode

	@property
	def global_frame(self):
		return self.global_step * self.cfg.action_repeat
	
	def get_depth(self, env):
		depth = env._env._env._env.render_depth()
		return depth

	def generate_eval(self):
		step, episode, total_reward = 0, 0, 0
		generate_until_episode = utils.Until(self.cfg.num_demos)
		observations_list = list()
		states_list = list()
		actions_list = list()
		rewards_list = list()
		depths_list = list()
		save_every = 1
		num_success = 0
		while num_success < self.cfg.num_demos:#generate_until_episode(episode):
			observations = list()
			states = list()
			actions = list()
			rewards = list()
			depths = list()
			episode_reward = 0
			i = 0
			time_step = self.eval_env.reset()
			depth = self.get_depth(self.eval_env)
			self.video_recorder.init(self.eval_env)
			goal_achieved = False
			while not time_step.last():
				if i % save_every == 0:
					observations.append(time_step.observation['pixels'])
					states.append(time_step.observation['features'])
					rewards.append(time_step.reward)
					depths.append(depth)
				with torch.no_grad(), utils.eval_mode(self.agent):
					action = self.agent.act(
						time_step.observation['pixels'],
						self.global_step,
						eval_mode=True)
				time_step = self.eval_env.step(action)
				depth = self.get_depth(self.eval_env)
				if i % save_every == 0:
					actions.append(time_step.action)
					self.video_recorder.record(self.eval_env)
				total_reward += time_step.reward
				episode_reward += time_step.reward
				step += 1
				i = i + 1
				if time_step.observation['goal_achieved'] > 0.5:
					goal_achieved = True
					break
			print(episode, episode_reward, goal_achieved)
			if goal_achieved:#episode_reward > 100:
			#if episode_reward > 100:
				episode += 1
				self.video_recorder.save(f'{episode}_eval.mp4')
				rewards_list.append(np.array(rewards))
				observations_list.append(np.stack(observations, 0))
				states_list.append(np.stack(states, 0))
				actions_list.append(np.stack(actions, 0))
				depths_list.append(np.stack(depths, 0))
				num_success += 1
				
		# Make np arrays
		for i in range(len(observations_list)):
			print("write i")
			observations_save = observations_list[i][None, :]
			states_save = states_list[i][None, :]
			actions_save = actions_list[i][None, :]
			rewards_save = rewards_list[i][None, :]
			depths_save = depths_list[i][None, :]

			# Save demo in pickle file
			save_dir = Path(self.work_dir)
			save_dir.mkdir(parents=True, exist_ok=True)
			snapshot_path = save_dir / f'expert_demos_{i}.pkl'
			payload = [
				observations_save, states_save, actions_save, rewards_save, depths_save
			]

			with open(str(snapshot_path), 'wb') as f:
				pickle.dump(payload, f)

	def load_snapshot(self, snapshot):
		with snapshot.open('rb') as f:
			payload = torch.load(f)
		agent_payload = {}
		for k, v in payload.items():
			if k not in self.__dict__:
				agent_payload[k] = v
		self.agent.load_snapshot(agent_payload)


@hydra.main(config_path='cfgs', config_name='config_generate')
def main(cfg):
	from generate import Workspace as W
	root_dir = Path.cwd()
	workspace = W(cfg)

	# Load weights	
	snapshot = Path(cfg.weight)
	if snapshot.exists():
		print(f'resuming: {snapshot}')
		workspace.load_snapshot(snapshot)
	
	workspace.generate_eval()


if __name__ == '__main__':
	main()
