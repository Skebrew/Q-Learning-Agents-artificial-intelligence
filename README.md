Steven Grossman's home work assignment from the University of Iowa's artificial intelligence class.
My contributions where in analysis.py, qlearningAgents.py, and valueIterationAgents.py. The rest of the code was provided by Bijaya Adhikari from the University of Iowa. All Q-learning agents use the same Q-Learning code created by Steven Grossman.

In order to run this program, you be using windows and have python installed so that you are able to run python from the command line.
To run:
1. Download the zipped project file by clicking the green button that says "<> Code" then click "download ZIP"
2. Unzip the file and then open a command line terminal inside the extracted folder. 
3. Enter any of the commands below to run the agent.


The agents created use Q-Learning with Q-tables to learn which action is best under each state. 
The crawler agent will maximize moving to the right.
run ```python crawler.py``` or ```python3 crawler.py``` 

There are several perameters to mess with to get the agent to behave differently. 
Epsilon is the chance that the agent will choose its next action randomly. 
At 0 the agent decides each of its moves and at 1 the agent acts completely randomly.

Step Delay is just for the animation. Set it to a really low value if you want it to learn quicker.

Learning rate is the amount of time it takes for the agent to forget one of its past experiences. 
Set to 0 and it will stop learning, set to 1 and it will completely replace what it has learned with its most recent experience.

Discount is how highly the agent values immidiat rewards and shouldn't be altered from its default value.

To see it quickly learn, first, set the step delay to 0, then turn epsilon all the way to 0.999. Then lower epsilon. The longer you take to lower epsilon, the faster it will be. Next, lower the learning rate down to 0. Finally raise step delay back up to 0.10000 and watch how fast it crawls!


The pacman agent will learn to minimize ghost contact and maximize pellets eaten. You can observe the very beginning of his training by running: 
```python pacman.py -p PacmanQAgent -n 10 -l smallGrid -a numTraining=10``` or   
```python3 pacman.py -p PacmanQAgent -n 10 -l smallGrid -a numTraining=10```

To see how efficient Pacman becomes at solving the maze, run 
```python autograder.py -q q5``` or ```python3 autograder.py -q q5```

The console will show pacman's statistics from training 2000 times, then learning will be disabled and Pacman will use its past knowledge to win the smallGrid maze. 
Pacman's score is based on pellets eaten, how long it takes for him to complete the maze, and whether he wins or loses.
