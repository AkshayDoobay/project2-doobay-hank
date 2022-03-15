# multiAgents.py
# --------------
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


from operator import countOf
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        foodList = newFood.asList()
        foodDistance = [manhattanDistance(newPos, food) for food in foodList]

        if len(foodDistance) == 0:
            return 0

        x = 0

        if action == 'Stop':
            x = -20

        ghostPos = newGhostStates[0].getPosition()
        ghostDis = manhattanDistance(newPos, ghostPos)

        return childGameState.getScore() + ghostDis / min(foodDistance) + x


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def MimiMaxChoice(self, gameState, agentIndex, depth):
        # if win, lose, or reached max depth return evaluation of current state
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        # current state is pacman, find max of all actions
        if agentIndex == 0:
            actions = [self.MimiMaxChoice(gameState.getNextState(agentIndex, action), agentIndex + 1, depth) for
                       action in gameState.getLegalActions(agentIndex)]
            return max(actions)
        # current state is ghost, find min of all actions
        # don't need to switch back to pacMan on next iteration of minimax
        elif agentIndex != gameState.getNumAgents() - 1:
            actions = [self.MimiMaxChoice(gameState.getNextState(agentIndex, action), agentIndex + 1, depth) for
                       action in gameState.getLegalActions(agentIndex)]
            return min(actions)
        # current state is ghost, find min of all actions
        # switch back to pacman on next iteration by resetting state to 0
        else:
            actions = [self.MimiMaxChoice(gameState.getNextState(agentIndex, action), 0, depth + 1) for action in
                       gameState.getLegalActions(agentIndex)]
            return min(actions)

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        nextActions = gameState.getLegalActions(0)
        maxVal = -10000
        bestAction = None

        for action in nextActions:
            if self.MimiMaxChoice(gameState.getNextState(0, action), 1, 0) > maxVal:
                maxVal = self.MimiMaxChoice(gameState.getNextState(0, action), 1, 0)
                bestAction = action

        return bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def max_value (self, gameState, depth, alpha, beta):
       

        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        value = -1000000
        for action in gameState.getLegalActions(0):
            value = max(value, self.min_value(gameState.getNextState(0, action), 1, depth, alpha, beta))
            # pruning
            if (value > beta):
                return value

            alpha = max(alpha, value)

        return value


    def min_value (self, gameState, agentIndex, depth, alpha, beta):
        

        if gameState.isWin() or gameState.isLose() or depth == self.depth: 
            return self.evaluationFunction(gameState)

        value = 1000000
        for action in gameState.getLegalActions(agentIndex):
            if agentIndex != gameState.getNumAgents() - 1:
                value = min(value, self.min_value(gameState.getNextState(agentIndex,action), agentIndex + 1, depth, alpha, beta))
            # switch back to pacman with a new depth
            else:  
                value = min(value, self.max_value(gameState.getNextState(agentIndex, action), depth + 1, alpha, beta))
            # pruning
            if (value < alpha):
                return value
                
            beta = min(beta, value)

        return value

    

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """

        alpha = -100000000 
        beta = 100000000

        value = -100000000
        bestAction = None
        for action in gameState.getLegalActions(0):
            value = self.min_value(gameState.getNextState(0, action), 1, 0, alpha, beta)
            if (alpha < value):
                alpha = value
                bestAction = action

        return bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def ExpectimaxChoice(self, gameState, agentIndex, depth):
        # if win, lose, or reached max depth return evaluation of current state
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        # current state is pacman, find max of all actions
        if agentIndex == 0:
            actions = [self.ExpectimaxChoice(gameState.getNextState(agentIndex, action), agentIndex + 1, depth) for
                       action in gameState.getLegalActions(agentIndex)]
            return max(actions)
        # current state is ghost, find average of all actions
        # don't need to switch back to pacMan on next iteration of minimax
        elif agentIndex != gameState.getNumAgents() - 1:
            actions = [self.ExpectimaxChoice(gameState.getNextState(agentIndex, action), agentIndex + 1, depth) for
                       action in gameState.getLegalActions(agentIndex)]
            return sum(actions) / len(gameState.getLegalActions(agentIndex))
        # current state is ghost, find average of all actions
        # switch back to pacman on next iteration by resetting state to 0
        else:
            actions = [self.ExpectimaxChoice(gameState.getNextState(agentIndex, action), 0, depth + 1) for action in
                       gameState.getLegalActions(agentIndex)]
            return sum(actions) / len(gameState.getLegalActions(agentIndex))

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        nextActions = gameState.getLegalActions(0)
        maxVal = -10000
        bestAction = None

        for action in nextActions:
            if self.ExpectimaxChoice(gameState.getNextState(0, action), 1, 0) > maxVal:
                maxVal = self.ExpectimaxChoice(gameState.getNextState(0, action), 1, 0)
                bestAction = action

        return bestAction


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    ghost_pos = currentGameState.getGhostPositions()
    pacman_pos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    heuristic = 0

    # add up distance from each food position
    for food in foodList:
        heuristic += manhattanDistance(pacman_pos, food)

    # add up distance from each ghost
    # if smaller than zero -- default to zero
    for ghost in ghost_pos:
        dist = max(4 - manhattanDistance(pacman_pos, ghost), 0)
        heuristic += dist

    # add random number to heuristic to satisfy cases where two actions are equally desired
    heuristic += random.randint(-5, 5)

    # super entice ghost if one option results in winning the game
    gameWin = 0
    if currentGameState.isWin():
        gameWin += 10000

    return currentGameState.getScore() - heuristic + gameWin


# Abbreviation
better = betterEvaluationFunction
