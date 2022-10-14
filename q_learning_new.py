import gym
import random
import numpy as np

def printway(index):
    value = ""
    if index==0:
        value = "down"
    elif index==1:
        value ="up"
    elif index==2:
        value ="right"
    elif index==3:
        value ="left"
    elif index==4:
        value ="pickup passenger"
    elif index==5:
        value ="drop off passenger"
    return value

streets = gym.make("Taxi-v3",render_mode="human").env
observation, info = streets.reset(seed=42)
initial_state = streets.encode(2, 3, 2, 0)
streets.s = initial_state
streets.render()

q_table = np.zeros([streets.observation_space.n, streets.action_space.n])

def q_learning_new(epochs,exploration,discount_factor,learning_rate):
    length = 0
    for taxi_run in range(epochs):
        state, info = streets.reset()
        done = False
        print("taxi run no is " + str(taxi_run) + "\n")
        i = 0
        while not done:
            random_value = random.uniform(0, 1)
            if (random_value < exploration): # explore
                action = streets.action_space.sample()  # Explore a random action
            else: # exploit
                action = np.argmax(q_table[state])  # Use the action with the highest q-value
                i += 1
                print("run no: "+str(taxi_run)+" | step " + str(i) + " " + printway(action))

            next_state, reward, done, _, info = streets.step(action)
            prev_q = q_table[state, action]
            next_max_q = np.max(q_table[next_state])
            new_q = (1 - learning_rate) * prev_q + learning_rate * (reward + discount_factor * next_max_q)
            q_table[state, action] = new_q
            state = next_state
            if (done == True):
                print("done is true.. episode ended \n")
                print("total steps: " + str(i) + "\n")
                length+=i
    avglen = length / epochs
    np.save("data",q_table)
    return avglen
