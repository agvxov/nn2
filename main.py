from os import getcwd, chdir
from time import sleep
from random import randint
from block2d import Block2d
from pyray import *

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env
from stable_baselines3 import *

screen_width = 800
screen_height = 600

class BlockGameEnvironment(gym.Env):
	metadata = {
		"render_modes": ["human"],
		"render_fps": 20,
	}

	def __init__(self, render_mode=None):
		#super().__init__()
		# Raylib
		init_window(screen_width, screen_height, "Physics Simulation")
		self.frame_counter = 0
		self.content = Block2d()

		# Action space
		#  with normalization
		self.action_space = spaces.Box(
			low=np.array([-1, -1, -1]),
			high=np.array([1, 1, 1]),
			dtype=np.float32 
		)
		#  without normalization
		#self.action_space = spaces.Box(
		#	low=np.array([0, 0, 0]),
		#	high=np.array([screen_width, screen_width, screen_height]),
		#	dtype=np.float32 
		#)
		
		# Observation space
		self.observation_space = spaces.Box(
			low=np.array([0] * 6),
			high=np.array([
				screen_width,
				screen_height,
				screen_width,
				screen_height,
				#self.content.max_speed,
				#self.content.max_speed,
				1,
				1,
			]),
			dtype=np.float64
		)
	
	def reset(self, seed=None, options=None):
		super().reset(seed=seed)

		# Reconstruct the game
		del self.content

		self.box_start     = (randint(0, screen_width), randint(0, screen_height))
		self.goal_position = (randint(0, screen_width), randint(0, screen_height))
		self.content = Block2d(
			Vector2(*self.box_start),
			Vector2(*self.goal_position),
		)

		obs = np.array([
			self.content.position.x,
			self.content.position.y,
			self.content.goal_position.x,
			self.content.goal_position.y,
			#self.content.speed.x,
			#self.content.speed.y,
			1,
			1,
		])

		return obs, {}
	
	def step(self, action):
		def calc_reward(is_done : bool) -> float:
			# We have many strategies to reward the model,
			#  one worse than the other.
			def keep_moving_large_proximity_big_earner():
				if is_done: return 100
				diff = (
					abs(self.content.goal_position.x - action[0]),
					abs(self.content.goal_position.y - action[1]),
				)
				if diff[0] < 100 and diff[1] < 100:
					return 1 / (diff[0] + diff[1])
				if self.content.speed.x == 0 and self.content.speed.y == 0:
					return -0.001
				return -0.0001
			def precise_proximity_punisher():
				r = 0
				diffs = (
					self.content.goal_position.x - self.content.position.x,
					self.content.goal_position.y - self.content.position.y
				)
				diffs = (abs(diffs[0]), abs(diffs[1]))
				speed_sum = self.content.speed.x + self.content.speed.y
				r -= (diffs[0] + diffs[1]) / 1000
				r -= speed_sum / 1200
				return r
			def take_it_slow_large_proximity_big_earner():
				if is_done: return 100
				elif (
					self.content.speed.x < 100 and
					self.content.speed.y < 100 and
					abs(self.content.position.x - self.content.goal_position.x) < 50 and
					abs(self.content.position.y - self.content.goal_position.y) < 50
				): return 1
				else: return -0.0001
			def just_move_near_it_im_begging_you():
				r = 0
				diffs = (
					self.content.goal_position.x - self.content.position.x,
					self.content.goal_position.y - self.content.position.y
				)
				dist = np.sqrt(diffs[0]**2 + diffs[1]**2)
				r = dist / 10_000
				return -r
			def just_move_near_it_im_begging_you_normalized():
				if is_done: return 10
				r = 0
				max_dist = np.sqrt(screen_width**2 + screen_height**2)
				diffs = (
					self.content.goal_position.x - self.content.position.x,
					self.content.goal_position.y - self.content.position.y
				)
				dist = np.sqrt(diffs[0]**2 + diffs[1]**2)
				r = dist / max_dist
				return -r
			return just_move_near_it_im_begging_you_normalized()

		# normalization
		#should_move = action[0] > 0.8
		click = Vector2(action[1], action[2])
		click.x = ((click.x + 1) * screen_width) / 2
		click.y = ((click.y + 1) * screen_height) / 2
		should_move = True
		#click = Vector2(action[1], action[2])

		if should_move:
			self.content.control(click)

		self.content.update()

		obs = np.array([
			self.content.position.x,
			self.content.position.y,
			self.content.goal_position.x,
			self.content.goal_position.y,
			#self.content.speed.x,
			#self.content.speed.y,
			1,
			1,
		])

		is_done = self.content.is_win_condition()
		reward = calc_reward(is_done)

		# printf debugging
		#print(f"obs: {obs}")
		#print(f"act: {action}")
		#print(f"rew: {reward}")

		return obs, reward, is_done, False, {}

	def render(self, mode="human"):
		begin_drawing()
		clear_background(RAYWHITE)
		self.content.display()
		end_drawing()

		if self.frame_counter < 400: take_screenshot(f"frame_{self.frame_counter:05d}.png")
		else: sleep(0.01)

		self.frame_counter += 1
	
	def close(self):
		close_window()



# Env init
gym.envs.registration.register(
	id="BlockGameEnvironment-v0",
	entry_point=__name__+":BlockGameEnvironment",
)
env = gym.make("BlockGameEnvironment-v0", render_mode="human")
# very useful check, however it clones the environment,
#  which is bad in our case because raylib uses global data
#check_env(env.unwrapped)



# Model init
def get_mymodel(name : str, do_train : bool):
	model_name = "custom_model_-_sorta_works"
	model = None
	match name:
		case "myPPO":
			model = PPO(
				"MlpPolicy",
				env,
				learning_rate=0.001,
				ent_coef=0.1,
				batch_size=2, # NOTE: this is why we dont marge the two cases
				verbose=1,
				tensorboard_log="logs/",
			)
		case "myA2C":
			model = A2C( # really, really fast to train
				"MlpPolicy",
				env,
				verbose=1,
				tensorboard_log="logs/",
			)
	if do_train:
		for i in range(5000): model.learn(100)
		model.save(model_name)
	else:
		match name:
			case "myPPO":
				model = PPO.load(model_name, env=env)
			case "myA2C":
				model = A2C.load(model_name, env=env)
	return model

model = get_mymodel(
#	"myPPO",
	"myA2C",
#	True,
	False,
)

# Show what the model learned
while True:
	obs, _ = env.reset()
	for _ in range(1000):
		action, _ = model.predict(obs)
		obs, _, done, _, _ = env.step(action)
		print(action)
		env.render()
		if done: break
