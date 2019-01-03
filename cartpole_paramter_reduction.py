#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 18:33:49 2018

@author: Shakti
"""
import os
import random 
import gym
import numpy as np 
from collections import deque 
from keras.models import Sequential 
from keras.layers import Dense
from keras.optimizers import Adam 

env = gym.make('CartPole-v0')

state_size =env.observation_space.shape[0]-2 # i realised that this objective of 199 frames is short enough that
		# that falling off the screen is not really a limiting factor so i reduced the feature space to 2D
action_size = env.action_space.n

batch_size= 32 

n_episodes =100 # somewhat arbitrary - it has never taken this long to learn to play 

output_dir = 'model_output/cartpole'

if not os.path.exists(output_dir):
	os.makedirs(output_dir)

class DQNAgent :
	def __init__(self,state_size,action_size):
		self.state_size = state_size
		self.action_size =action_size
		self.memory = deque (maxlen=2000)
		self.gamma = 0.95
		self.epsilon =1.0
		self.epsilon_decay = 0.9
		self.epsilon_min =0.01
		self.learning_rate =0.004 #0.001
		self.model = self._build_model()

	def _build_model(self):

		model = Sequential()
		model.add(Dense(24,input_dim=self.state_size,activation ='relu'))
		model.add(Dense(24,activation ='relu'))
		model.add(Dense(self.action_size,activation ='linear'))
		model.compile(loss='mse',optimizer = Adam(lr=self.learning_rate))

		return model 

	def store(self, state, action, reward, next_state,done):
		self.memory.append((state, action, reward, next_state,done))

	def act(self,state):
		if np.random.rand() < self.epsilon:
			if self.epsilon > self.epsilon_min:
				self.epsilon *= self.epsilon_decay

			return random.randrange(self.action_size)
		act_values = self.model.predict(state)
		if self.epsilon > self.epsilon_min:
				self.epsilon *= self.epsilon_decay


		return np.argmax(act_values[0])


	def update_model_weights(self, batch_size):
		minibatch = random.sample(self.memory,batch_size)

		for state, action, reward, next_state,done in minibatch :
			target = reward 
			if not done :
				target = reward + self.gamma * np.amax (self.model.predict(next_state)[0])

			target_f = self.model.predict(state)
			target_f[0][action] = target 

			self.model.fit (state,target_f,epochs =1 , verbose =0 )

		#if self.epsilon > self.epsilon_min:
			#self.epsilon *= self.epsilon_decay

	def load (self,name):
		self.model.load_weights(name)

	def save(self,name):
		self.model.save_weights(name)

agent = DQNAgent(state_size, action_size)	
#agent.load('model_output/cartpoleweights_reduced_param.hdf5')
#agent.load('model_output/cartpoleweights_average15.hdf5')

#agent.load('model_output/cartpoleweights_average15.hdf5')

#agent.load('cartpoleweights_best.hdf5')


train =True
test_model = True 
#lr =0.001, temp_eps=0.2 , threshold,30
#lr =0.004, temp_eps=0.2 , threshold,30
#lr =0.004, temp_eps=0.2 , threshold,10
#lr =0.004, temp_eps=0.2 , threshold,5

if train :

	
	prev_scores  =deque(maxlen=5)
	prev_scores.append(0)
	current_average = 0 
	prev_best_average  = 0

	temp_eps =0.1
	for e in range (n_episodes):
	
		agent.epsilon =temp_eps
		done = False 

		state = env.reset()[2:] # i realised that this objective of 199 frames is short enough that
		# that falling off the screen is not really a limiting factor so i reduced the feature space to 2D
		state = np.reshape(state, [1,state_size]) 
		
		for  time in range (201):
			#env.render()   # can uncomment if you would like to see model try and fail as it learns

			if(current_average> 5):
				temp_eps =0.005

			action = agent.act(state)

			next_state, reward,done, _ = env.step(action)
			next_state= next_state[2:]
			if abs(next_state[0]) >0.05 :
				done =True
				print('fail')
				print(next_state[0])



			next_state = np.reshape(next_state,[1,state_size])

			


			agent.store( state, action, reward, next_state,done)

			state = next_state

			if (max (prev_scores ) < 1.6 * current_average):
				if (time > max (prev_scores ) *0.9 and time < min(max (prev_scores ),max (prev_scores ) *0.7+45)):
					agent.epsilon=temp_eps*10

			

			if done: 
				print("episode: {}/{},score: {},e: {:2}".format(e,n_episodes,time,agent.epsilon))
				

				prev_scores.append(time)
				current_average = sum(prev_scores)/len(prev_scores)

				

				if (current_average >= prev_best_average):
					
					prev_best_average = current_average
					agent.save(output_dir + "_weights_" + 'reduced_param' + ".hdf5" )

				print('average : ', current_average , 'best average:  ', prev_best_average )

	
				break 
				
			if len(agent.memory) > batch_size:
				agent.update_model_weights(batch_size)

		if prev_scores[-1] >197:
			print ('trained for ' , e , 'episodes ')
			break 

			

if test_model:
	agent.load('model_output/cartpole_weights_reduced_param.hdf5')
	sum=0
	
	for e in range (100):
		done = False 

		state = env.reset()[2:]
		state = np.reshape(state, [1,state_size]) 
		agent.epsilon =0.0
		
		agent.epsilon_decay=0
		
		for  time in range (201):
			env.render()
			
			#state[0][3] = np.sign(state[0][2])*min(abs(state[0][2]),0.1)
			action = agent.act(state)

			next_state, reward,done, _ = env.step(action)


			#reward = reward if not done else -10 
			next_state= next_state[2:]
			if abs(next_state[0])>0.05:
				next_state[0] = np.sign(next_state[0])* 0.05

			next_state = np.reshape(next_state,[1,state_size])

			#agent.store( state, action, reward, next_state,done)

			state = next_state
			#print (state)

			if done: 
				
				print("episode: {}/{},score: {},e: {:2}".format(e+1,100,time,agent.epsilon))
				sum+=time
				break 
				
			
			
	print('average score: ' , sum/100)

