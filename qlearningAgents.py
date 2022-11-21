# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
from backend import ReplayMemory

import nn
import model
import backend
import gridworld


import random,util,math
import numpy as np
import copy

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent
      Functions you should fill in:     q_freq_table [state, action, Q-val, frequency]
        - computeValueFromQValues       (max a) Q(state, action)
        - computeActionFromQValues      returns argmax action of a state
        - getQValue                     Q(state, action)
        - getAction                     returns either argmax action of a state or a random action
        - update                        Q(s,a) <- Q(s,a) + a[R(s,a,s') + gamma(max a)Q(s',a') - Q(s,a)]
      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)
      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        "*** YOUR CODE HERE ***"
        # Need Q table for storing state-action pairs throught the table. Index by state. Initialize to 0   [state, action, Q-val, frequency]
        self.q_freq_table = util.Counter()
    # Q(state, action)
    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        # [Potential Bug] I'm assuming that I can get a Q-value straight from the q_table
        # [Potential Bug] This hasn't been tested yet and I don't know if you can index a table by two values
        if not isinstance(self.q_freq_table[state, action], int):   #checks to see if entry exists
            return self.q_freq_table[state, action][2]
        else:
            self.q_freq_table[state,action] = [state, action, 0, 0] #if the entry doesn't exist this adds it
            return self.q_freq_table[state,action][2]

    # (max a) Q(state, action)
    def computeValueFromQValues(self, state):   #NEEDS TESTING
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        """Important: Make sure that in your computeValueFromQValues and computeActionFromQValues functions, you only 
        access Q values by calling getQValue . This abstraction will be useful for question 6 when you override 
        getQValue to use features of state-action pairs rather than state-action pairs directly.                     """

        #(max action a)Q(state,action)
        # (max action a)
        actions = self.getLegalActions(state)
        score_choices = []
        if len(actions) > 0:
            for act in actions:
                val = self.getQValue(state, act)
                score_choices.append(val)
            return max(score_choices)
        else:
            return 0.0

    # returns argmax aprime of a state
    def computeActionFromQValues(self, state):  #NEEDS TESTING
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        """Important: Make sure that in your computeValueFromQValues and computeActionFromQValues functions, you only 
                access Q values by calling getQValue . This abstraction will be useful for question 6 when you override 
        getQValue to use features of state-action pairs rather than state-action pairs directly.                     """
        """Note: For computeActionFromQValues, you should break ties randomly for better behavior. The random.choice() 
                function will help. In a particular state, actions that your agent hasn’t seen before still have a Q-value, 
                specifically a Q-value of zero, and if all of the actions that your agent has seen before have a negative 
        Q-value, an unseen action may be optimal.                                                                    """

        # Look at all the values in a state.
        # return the action that corresponds with the highest score
        # in the event of a tie of scores, you should pick an arbitrary action to return
        # if all other Q-values are negative and one Q-value is 0, in this condition you should choose the zero score action
        #special circumstance where all other states are negative and one is zero

        actions = self.getLegalActions(state)

        # ZERO NEGATIVE BLOCK [Potential Bug] NEEDS TESTING
        is_negative_zero = True # starts true, becomes false when any positive q val is detected
        # [action] for storing any actions with a q value of zero
        zeroes = []
        #check to see if all scores are either negative or zero
        #if they are, return the one zero value
        # [Potential Bug] this is untested and removable, remove this code block if its too much work
        for act in actions:
            if self.getQValue(state, act) > 0:
                is_negative_zero = False
            elif self.getQValue(state, act) == 0:
                zeroes.append(act)
        if (is_negative_zero == True) and (len(zeroes) == 1):
            return zeroes[0]

        #calculate highest scoring action NEEDS TESTING
        # in the event of a tie of scores, you should pick an arbitrary action to return
        # [score, action]
        score_action_choices = []
        #[action]
        top_scorerers = []
        if len(actions) == 0:   # if we are in a terminal state return none
            return None
        for act in actions:
            curr_score = [self.getQValue(state, act), act]
            score_action_choices.append(curr_score)
        #gather the top scores into one place
        top_score = max(score_action_choices, key=lambda x:x[0])    #lambda makes max only look at first value
        for act in actions: # grab any actions that share the max score
            if self.getQValue(state, act) == top_score[0]:
                top_scorerers.append(act)
        if len(top_scorerers) == 1:
            return top_scorerers[0]     #[Potential Bug] I have not tested this return it may cause graphical errors
        else:
            rnd_choice = random.choices(top_scorerers)[0]   #random technically returns a list so make sure to get the first element [Fixed Bug]
            return rnd_choice

    # argmax aprime of a state Until Question 4
    def getAction(self, state): #NEEDS TESTING
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        "*** YOUR CODE HERE ***"
        # in the event of a tie of scores, you should pick an arbitrary action to return                                #for question 4 #NEEDS TESTING
        # flip a coin with probability epsilon, if tails return the action that corresponds with the highest score
        # if heads then choose a random action
        actions = self.getLegalActions(state)

        #proves that flipcoin is working [Proven]
        #heads = 0
        #tails = 0
        #for i in range(0,50):
        #    if util.flipCoin(self.epsilon):
        #        heads += 1
        #    else:
        #        tails += 1

        if util.flipCoin(self.epsilon):
            rand_choice = random.choices(actions)[0]
            return rand_choice

        #REPLACE THIS FOR QUESTION 4 ALSO THIS NEEDS TESTING
        ans = self.computeActionFromQValues(state)
        return ans
        #return self.getLegalActions(state)[0]


    def update(self, state, action, nextState, reward): #NEEDS TESTING
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # Q(s,a) <- Q(s,a) + a[R(s,a,s') + gamma(max a)Q(s',a') - Q(s,a)]

        #input: current state-action, reward from said action, next state
        #Need Q table for storing state-action pairs throught the table. Index by state. Initialize to 0
        #Need Nsa which is a table of frequencies visited for state-action pairs again initial 0 and index by state

        #initialize the entry on q_freq_table if it isn't already there and get its Q value [Potential Bug] for question 6 make sure that entries in the table are getting grabbed correctly
        #if current state-action exists [Potential Bug] I'm assuming I'm always given a valid value and it'll never be null
        current_val = self.getQValue(state, action)

        #increment Nsa(current state-action)
        self.q_freq_table[state, action][3] += 1

        #Q(current state-action) = Q(current state-action) + learning-rate(Nsa(current state-action))(reward + gamma(max action a')Q(next state-action) - Q(current state-action))
        #Q(current state-action) = Q(current state-action) + learning-rate(reward + gamma(max action a')Q(next state-action) - Q(current state-action))
        current_val = self.getQValue(state, action)                                                 #Q(current state-action)
        max_action_value_next_state = self.computeValueFromQValues(nextState)                       #(max action a')Q(next state-action)
        gamma_max_value_next_state = self.discount * max_action_value_next_state                    #gamma(max action a')Q(next state-action)
        reward_plus_gamma_next_state = reward + gamma_max_value_next_state                          #reward + gamma(max action a')Q(next state-action)
        reward_gamma_difference = reward_plus_gamma_next_state - current_val                        #(reward + gamma(max action a')Q(next state-action) - Q(current state-action))
        #frequency_of_visit = self.q_freq_table[state, action][3]                                   #Nsa(current state-action) [Potential Bug] NEEDS TESTING
        #learn_frequency_difference = self.alpha * frequency_of_visit * reward_gamma_difference     #learning-rate(Nsa(current state-action))(reward + gamma(max action a')Q(next state-action) - Q(current state-action))
        learn_frequency_difference = self.alpha * reward_gamma_difference                           #learning-rate(reward + gamma(max action a')Q(next state-action) - Q(current state-action))
        ans = current_val + learn_frequency_difference
        self.q_freq_table[state, action][2] = ans

        #current state-action <- next state, (argmax a')f( Q(next state-action), Nsa(next state, next action))
        #return None # Do we need to return anything?

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action

class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent
       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        #[index,
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action): #needs testing
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        waights = self.getWeights()
        feature_values = util.Counter()
        feats = self.featExtractor.getFeatures(state, action)
        #incase there are weights that are missing      [Potential Bug] do you initialize the weights to a random value?
        if len(waights) != len(self.featExtractor.getFeatures(state, action)):
            weights_to_initialize = len(self.featExtractor.getFeatures(state, action)) - len(waights)
            for i in range(0, weights_to_initialize):
                self.weights[i] = random.random()

        i = 0
        for feat in feats:
            feature_values[i] = feats[feat]
            i += 1
        ans = waights * feature_values  #computes the dot product of the weights and the feature values effectively finding the Q value [Potential Bug] this has not been tested
        return ans

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        #for each weight i:
            #new weighti val = old weighti val + (learning rate * difference * feature function i)
            #where difference = (reward + gamma(max val from actions)(Q'val of next state action)) - Q val of current state

        waights = self.getWeights()
        features = util.Counter()
        i = 0
        for feat in self.featExtractor.getFeatures(state, action):
            features[i] = self.featExtractor.getFeatures(state, action)[feat]
            i += 1

        # [Potential Bug] difference calculation was directly lifted from other q learning agent it may not be correct as it NEEDS TESTING
        new_waights = []
        i = 0
        for waight in waights:
            wi = waights[waight]    #old weighti val
            fi = features[i]
            learning_rate = self.alpha
            current_val = self.getQValue(state, action) #Q(current state action)
            max_action_value_next_state = self.computeValueFromQValues(nextState)                       #(max val from actions)(Q'val of next state action)
            gamma_max_value_next_state = self.discount * max_action_value_next_state                    #gamma(max val from actions)(Q'val of next state action)
            reward_plus_gamma_next_state = reward + gamma_max_value_next_state                          #reward + (max val from actions)(Q'val of next state action)
            difference = reward_plus_gamma_next_state - current_val                                     #(reward + gamma(max action a')Q(next state-action) - Q val of current state)
            # new weighti val = old weighti val + (learning rate * difference * feature function i)
            featurei_difference = difference * fi
            learning_rate_featurei_distance = featurei_difference * learning_rate
            new_weight = wi + learning_rate_featurei_distance
            new_waights.append(new_weight)  #creates a list of new weights to debug weights not updating correctly [Potential Bug]
            self.weights[i] = new_weight
            i += 1




    def final(self, state):
        """Called at the end of each game."""
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
