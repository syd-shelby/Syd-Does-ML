#i built this from an online tutorial https://medium.com/coinmonks/build-your-first-ai-game-bot-using-openai-gym-keras-tensorflow-in-python-50a4d4296687
#ideas for improvements
    #learn about its neural network model selection
    #build it myself using an easier classifier (NB?)
    #learn about one hot encoding
    #first step is random, but I think you can learn about initial parameters from the env.reset = state stuff
    #get rid of goal steps
import gym
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

env = gym.make("CartPole-v1")
state = env.reset()
step = 0
goal_steps = 500
score_requirement = 60
intial_games = 10000

#this essentially creates our training data. 
# we run the game 10000 times and keep results of the ones that reach the  score requirement
#the observations at each step of the successful run and the action associated with it are turned into the training data for the model

def model_data_prep():
    training_data = []
    accepted_scores = []
    
    for game_index in range(intial_games):
        score = 0
        game_memory = []
        previous_observation = []
        
        for step_index in range(goal_steps):
            action = random.choice(range(env.action_space.n))
            observation, reward, done, info = env.step(action)
            
            if len(previous_observation) > 0:
                game_memory.append([previous_observation, action])

            previous_observation = observation
            score += reward
            
            if done:
                break
        
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0,1]
                elif data[1] == 0:
                    output = [1,0]
                training_data.append([data[0], output])

        env.reset()

    print(accepted_scores)
    return training_data
 
#gross neural network stuff that I dont understand yet
#re write with a simple classifier????
def build_model(input_size, output_size):
    model = Sequential()
    model.add(Dense(128, input_dim=input_size, activation='relu'))
    model.add(Dense(52, activation='relu'))
    model.add(Dense(output_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam())

    return model

#train the above model using training data
def train_model(training_data):
    x = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]))
    y = np.array([i[1] for i in training_data]).reshape(-1,len(training_data[0][1]))
    model = build_model(input_size=len(x[0]), output_size=len(y[0]))

    model.fit(x,y,epochs=10)
    return model

trained_model = train_model(model_data_prep())

scores = []
choices = []
for each_game in range(100):
    score =0
    prev_obs = []
    for step_index in range(goal_steps):
        env.render()
        if len(prev_obs)==0:
                action = random.randrange(0,2)
        else:
            action = np.argmax(trained_model.predict(prev_obs.reshape(-1,len(prev_obs)))[0])

        choices.append(action)
        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        score+=reward
        if done:
            break

    env.reset()
    scores.append(score)

print(scores)
print('Average Score: ', sum(scores)/len(scores))

