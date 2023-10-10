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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # Calculate the minimum distance to a ghost
        minGhostDistance = min(manhattanDistance(newPos, state.getPosition()) for state in newGhostStates)

        # Calculate the distance to the nearest food
        pos = currentGameState.getPacmanPosition()
        nearestFoodDistance = min(manhattanDistance(pos, food) for food in currentGameState.getFood().asList())

        # Calculate the change in score
        scoreDiff = successorGameState.getScore() - currentGameState.getScore()

        # Calculate the change in distance to the nearest food
        newFoodsDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        newNearestFoodDistance = min(newFoodsDistances) if newFoodsDistances else 0
        isFoodNearer = nearestFoodDistance - newNearestFoodDistance

        # Keep direction to avoid meaningless random movements when criteria are not satisfied
        direction = currentGameState.getPacmanState().getDirection()

        # Reflex formula
        if action == Directions.STOP or minGhostDistance <= 1 :
            return 0
        elif scoreDiff > 0:
            return 8
        elif isFoodNearer > 0:
            return 4
        elif action == direction:
            return 2

        return 1


def scoreEvaluationFunction(currentGameState: GameState):
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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def findMinValue(state, agentIndex, depth):
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:  # No legal actions mean the game is finished (win or lose)
                return self.evaluationFunction(state)

            # When all ghosts have moved, it's Pacman's turn
            if agentIndex == state.getNumAgents() - 1:
                return min(findMaxValue(state.generateSuccessor(agentIndex, action), depth) for action in legalActions)
            else:
                return min(findMinValue(state.generateSuccessor(agentIndex, action), agentIndex + 1, depth) for action in legalActions)

        def findMaxValue(state, depth):
            legalActions = state.getLegalActions(0)
            if not legalActions or depth == self.depth:
                return self.evaluationFunction(state)

            return max(findMinValue(state.generateSuccessor(0, action), 0 + 1, depth + 1) for action in legalActions)

        bestAction = max(gameState.getLegalActions(0), key=lambda action: findMinValue(gameState.generateSuccessor(0, action), 1, 1))
        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        Infinity = float('inf')

        def minValue(state, agentIndex, depth, a, b):
            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state)

            v = Infinity
            for action in legalActions:
                newState = state.generateSuccessor(agentIndex, action)

                # Is it the last ghost?
                if agentIndex == state.getNumAgents() - 1:
                    newV = maxValue(newState, depth, a, b)
                else:
                    newV = minValue(newState, agentIndex + 1, depth, a, b)

                v = min(v, newV)
                if v < a:
                    return v
                b = min(b, v)
            return v

        def maxValue(state, depth, a, b):
            legalActions = state.getLegalActions(0)
            if not legalActions or depth == self.depth:
                return self.evaluationFunction(state)

            v = -Infinity
            # For enable second play pruning
            if depth == 0:
                bestAction = legalActions[0]
            for action in legalActions:
                newState = state.generateSuccessor(0, action)
                newV = minValue(newState, 0 + 1, depth + 1, a, b)
                if newV > v:
                    v = newV
                    if depth == 0:
                        bestAction = action
                if v > b:
                    return v
                a = max(a, v)

            if depth == 0:
                return bestAction
            return v

        bestAction = maxValue(gameState, 0, -Infinity, Infinity)
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def find_max_value(state, depth):
            legal_actions = state.getLegalActions(0)
            if not legal_actions or depth == self.depth:
                return self.evaluationFunction(state)

            max_value = max(find_expected_value(state.generateSuccessor(0, action), 0 + 1, depth + 1) for action in legal_actions)
            return max_value

        def find_expected_value(state, agent_index, depth):
            legal_actions = state.getLegalActions(agent_index)
            if not legal_actions:
                return self.evaluationFunction(state)

            probability = 1.0 / len(legal_actions)
            expected_value = 0
            for action in legal_actions:
                new_state = state.generateSuccessor(agent_index, action)
                if agent_index == state.getNumAgents() - 1:
                    expected_value += find_max_value(new_state, depth) * probability
                else:
                    expected_value += find_expected_value(new_state, agent_index + 1, depth) * probability
            return expected_value

        legal_actions = gameState.getLegalActions()
        best_action = max(legal_actions, key=lambda action: find_expected_value(gameState.generateSuccessor(0, action), 1, 1))
        return best_action

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacman_position = currentGameState.getPacmanPosition()
    food_positions = currentGameState.getFood().asList()

    closest_food_distance = min(manhattanDistance(pacman_position, food) for food in food_positions) if food_positions else 0.5
    current_score = currentGameState.getScore()

    evaluation = 1.0 / closest_food_distance + current_score
    return evaluation


# Abbreviation
better = betterEvaluationFunction
