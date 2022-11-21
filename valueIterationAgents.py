# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        """
        Note: A policy synthesized from values of depth k (which reflect the next k rewards) will actually reflect 
        the next k+1 rewards (i.e. you return ). Similarly, the Q-values will also reflect one more reward than the 
        values (i.e. you return Qk+1).
        You should return the synthesized policy pi(k+1)
        """

        #Start with V0(s) = 0: no time steps left means an expected reward sum of zero
        #start by making a counter initialized to all zeros (the default for a counter)
        #Hint: You may optionally use the util.Counter class in util.py, which is a dictionary with a default value of zero.
        #However, be careful with argMax: the actual argmax you want may be a key not in the counter!
        #this means that you should calculate the exit option as it will not be in the counter [Potential Bug] this may not be the only case

        #local variables include the old_table, new_table, and change. old_table is the table that has all the
        # values of the previous iteration. New table will be the table we update from values given by old_table,
        #and change will be the relative change from one iteration to the next. If there's zero change it means
        #that the values have converged.

        old_table = util.Counter()
        change = 1  #must start at high number so that the while loop functions
        all_states = self.mdp.getStates()
        for st in all_states:   #populates oldtable with all states
            if st == ('TERMINAL_STATE'):    #[Potential Bug] Hopefully by removing this I still consider all the statets
                print() #do nothing
            else:
                old_table[st[0], st[1]]
        new_table = old_table.copy()
        action_choices = []  # Holds eadh of the actions with their Q value [value, action]
        iteration_count = 0   #this will track which iteration we're in
        iteration_total = self.iterations #this will tell us how many times to run value iteration

        #Given vector of Vk(s) values, compute Vk+1(s) until the values converge
        # Note: Make sure to handle the case when a state has no available actions in an MDP (think about what this means for future rewards). [Potential Bug]
        # Vk+1(s) <- (max a) (sum s') T(s,a,s') [R(s,a,s') + (gamma * Vk(s'))]

        #old_table = new_table
        #change = 0
        #for state s in all states (use mdp.getStates())
        #new_table(s) = (max a) Qval
        #if |new_table(s) - old_table(s)| > change then change = |new_table(s) - old_table(s)|
        while iteration_total >= iteration_count:   # Switch this to change > 0 in order to run until convergence
            change = 0
            self.values = new_table.copy()
            old_table = new_table.copy()
            for stat in all_states:
                actions = self.mdp.getPossibleActions(stat)
                if actions.count('exit') != 0:  #this assumes we are forced to take the exit action on preterminal nodes [Potential Bug]
                    new_table[stat] = self.computeQValueFromValues(stat, 'exit')
                    converg_diff = abs((new_table[stat]) - (old_table[stat]))  # this is the difference between the old state and the new. Used for testing convergance
                    if (converg_diff > change):
                        change = converg_diff
                elif len(actions) > 0:
                    action_choices.clear()
                    for act in actions:
                        vl = self.computeQValueFromValues(stat, act)  #[Potential Bug] computeQValueFromValues hasn't been thoroughly tested
                        action_choices.append([vl, act]) #appends to choices the action along with its Q value
                    new_val = max(action_choices, key=lambda x:x[0])# contains the best action's value
                    new_table[stat] = new_val[0]
                    converg_diff = abs((new_table[stat]) - (old_table[stat]))   #this is the difference between the old state and the new. Used for testing convergance
                    if (converg_diff > change):
                        change = converg_diff
                else:
                    if not (self.mdp.isTerminal(stat)):
                        new_table[stat] = 0 #[Potential Bug] this could possibly not the correct answer the professor was talking about when he said
                        # Note: Make sure to handle the case when a state has no available actions in an MDP (think about what this means for future rewards).
                    #else:
            iteration_count += 1

        #get policy by calculating the best action per piece
        # pi(s) = argmax(a) (sum s') T(s,a,s') [R(s,a,s') + (gamma * Vk(s'))]
        policy_table = new_table.copy() # for holding the policy strings
        possible_actions = [] #for holding which actions have the best value for argmax a [Value, Action]
        for stat in new_table:
            actions = self.mdp.getPossibleActions(stat)
            if actions.count('exit') != 0:  # this assumes we are forced to take the exit action on preterminal nodes [Potential Bug]
                policy_table[stat] = 'exit'
            elif len(actions) > 0:
                action_choices.clear()
                for act in actions:
                    vl = self.computeQValueFromValues(stat, act)  # [Potential Bug] computeQValueFromValues hasn't been thoroughly tested
                    action_choices.append([vl, act])  # appends to choices the action along with its Q value
                new_val = max(action_choices, key=lambda x: x[0])  # contains the best action's value
                policy_table[stat] = new_val[1]
            else:
                if not (self.mdp.isTerminal(stat)):
                    policy_table[stat] = 'stuck'  # [Potential Bug] this could possibly not the correct answer the professor was talking about when he said
                    # Note: Make sure to handle the case when a state has no available actions in an MDP (think about what this means for future rewards).
                    #[Potential Bug] potentially None could be an invalid action. I don't know what to put if there is no action
                # else:
        #new_table  #turn this into a policy at the end

        return policy_table







    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # find Q*(s,a) = sum for all s' T(s,a,s')[R(s,a,s') + gammaV*(s')]
        act_q_val = 0 #sum s'
        # this gets all the possible states and probabilities that could result from taking action act at state
        sprimes = self.mdp.getTransitionStatesAndProbs(state, action)
        for next_state in sprimes:
            rwd = self.mdp.getReward(state, action, next_state)    #R(s,a,s')
            prob = next_state[1]    #T(s,a,s')
            #nexval = self.getValue(next_state[0])   #V*(s')
            nexval = self.getValue(next_state[0])   #V*(s')
            discounted_next_value = self.discount * nexval    #gammaV*(s')
            next_state_value = prob * (rwd + discounted_next_value) #T(s,a,s')[R(s,a,s') + gammaV*(s')]
            act_q_val += next_state_value   #sum s'
        return act_q_val
        #[Possible Bug] this code was directly ripped from computeActionFromValues and it is untested


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        action_choices = []  #will hold the [value of action, action] of the possible actions
        #Base case
        if self.mdp.isTerminal(state):
            return None
        # find max a Q*(s,a)
        else:
            actions = self.mdp.getPossibleActions(state)
            for act in actions:
                # Q* = sum s' T(s,a,s')[R(s,a,s') + gammaV*(s')]
                act_q_val = 0 #sum s'
                sprimes = self.mdp.getTransitionStatesAndProbs(state, act) # this gets all the possible states and probabilities that could result from taking action act at state
                for next_state in sprimes:
                    rwd = self.mdp.getReward(state, act, next_state)    #R(s,a,s')
                    prob = next_state[1]    #T(s,a,s')
                    discounted_next_value = self.discount * self.getValue(next_state[0])    #gammaV*(s')
                    next_state_value = prob * (rwd + discounted_next_value) #T(s,a,s')[R(s,a,s') + gammaV*(s')]
                    act_q_val += next_state_value   #sum s'
                action_choices.append([act_q_val, act]) #Q* value of action
            ans = max(action_choices, key=lambda x:x[0])    #lambda makes max only look at first value
            ans = ans[1]
            return ans

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
