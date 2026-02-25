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

        foodList = newFood.asList()

        foodBonus = 0.0
        foodCountPenalty = 0.0
        if foodList:
            minFoodDist = min(manhattanDistance(newPos, f) for f in foodList)
            foodBonus = 10.0 / (minFoodDist + 1)
            foodCountPenalty = 4.0 * len(foodList)

        ghostTerm = 0.0
        for i, ghostState in enumerate(newGhostStates):
            gpos = ghostState.getPosition()
            dist = manhattanDistance(newPos, gpos)

            if newScaredTimes[i] > 0:
                ghostTerm += 200.0 / (dist + 1)
            else:
                if dist <= 1:
                    ghostTerm -= 500
                else:
                    ghostTerm -= 2.0 / (dist + 1)

        stopPenalty = 10.0 if action == Directions.STOP else 0.0

        return successorGameState.getScore() + foodBonus - foodCountPenalty + ghostTerm - stopPenalty

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
        numAgents = gameState.getNumAgents()

        def minimax(state, depth, agentIndex):

            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            if agentIndex == 0:
                value = float("-inf")
                actions = state.getLegalActions(agentIndex)
                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value = max(
                        value,
                        minimax(successor, depth, 1)
                    )
                return value

            else:
                value = float("inf")
                actions = state.getLegalActions(agentIndex)

                nextAgent = agentIndex + 1
                nextDepth = depth

                if nextAgent == numAgents:
                    nextAgent = 0
                    nextDepth += 1

                for action in actions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value = min(
                        value,
                        minimax(successor, nextDepth, nextAgent)
                    )
                return value

        # Root decision (Pacman choosing action)
        bestValue = float("-inf")
        bestAction = None

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = minimax(successor, 0, 1)

            if value > bestValue:
                bestValue = value
                bestAction = action

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
        numAgents = gameState.getNumAgents()
        alpha = float("-inf")
        beta = float("inf")

        def abValue(state, agentIndex, pacmanPlies, alpha, beta):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            if agentIndex == 0 and pacmanPlies == self.depth:
                return self.evaluationFunction(state)

            actions = state.getLegalActions(agentIndex)
            if not actions:
                return self.evaluationFunction(state)

            nextAgent = (agentIndex + 1) % numAgents

            if agentIndex == 0:
                best = float("-inf")
                for action in actions:
                    succ = state.generateSuccessor(agentIndex, action)
                    nextPlies = pacmanPlies + (1 if nextAgent == 0 else 0)
                    val = abValue(succ, nextAgent, nextPlies, alpha, beta)

                    if val > best:
                        best = val

                    if best > beta:
                        return best
                    alpha = max(alpha, best)
                return best

            else:
                best = float("inf")
                for action in actions:
                    succ = state.generateSuccessor(agentIndex, action)
                    nextPlies = pacmanPlies + (1 if nextAgent == 0 else 0)
                    val = abValue(succ, nextAgent, nextPlies, alpha, beta)

                    if val < best:
                        best = val

                    if best < alpha:
                        return best
                    beta = min(beta, best)
                return best

        bestAction = Directions.STOP
        bestScore = float("-inf")

        for action in gameState.getLegalActions(0):
            succ = gameState.generateSuccessor(0, action)
            score = abValue(succ, 1 % numAgents, 0, alpha, beta)

            if score > bestScore:
                bestScore = score
                bestAction = action

            alpha = max(alpha, bestScore)

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
        numAgents = gameState.getNumAgents()

        def expectimax(state: GameState, agentIndex: int, ply: int):
            if state.isWin() or state.isLose() or ply == self.depth:
                return self.evaluationFunction(state)

            actions = state.getLegalActions(agentIndex)
            if not actions:
                return self.evaluationFunction(state)

            nextAgent = (agentIndex + 1) % numAgents
            nextPly = ply + 1 if nextAgent == 0 else ply

            if agentIndex == 0:
                bestVal = float("-inf")
                for a in actions:
                    succ = state.generateSuccessor(agentIndex, a)
                    bestVal = max(bestVal, expectimax(succ, nextAgent, nextPly))
                return bestVal

            prob = 1.0 / len(actions)
            expVal = 0.0
            for a in actions:
                succ = state.generateSuccessor(agentIndex, a)
                expVal += prob * expectimax(succ, nextAgent, nextPly)
            return expVal

        bestAction = Directions.STOP
        bestVal = float("-inf")

        for action in gameState.getLegalActions(0):
            succ = gameState.generateSuccessor(0, action)
            val = expectimax(succ, 1 % numAgents, 0) 
            if val > bestVal:
                bestVal = val
                bestAction = action

        return bestAction

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION (high level):
    - Start from the built-in score.
    - Encourage eating food: closer food => better.
    - Encourage clearing all food: fewer food left => better.
    - Encourage capsules: fewer capsules left => better, and closer capsule => better.
    - Ghosts:
        * If ghost not scared: heavily penalize being too close.
        * If ghost scared: reward being close (so you can eat it).
    """
    from util import manhattanDistance
    from game import Directions
    if currentGameState.isWin():
        return float("inf")
    if currentGameState.isLose():
        return float("-inf")

    pos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    capsules = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()

    score = currentGameState.getScore()

    if foodList:
        dFoodMin = min(manhattanDistance(pos, f) for f in foodList)
        score += 12.0 / (dFoodMin + 1.0)
        score -= 4.0 * len(foodList)
    else:
        score += 1000.0

    if capsules:
        dCapMin = min(manhattanDistance(pos, c) for c in capsules)
        score += 4.0 / (dCapMin + 1.0)
        score -= 18.0 * len(capsules)

    activeMin = None
    scaredMin = None
    totalScared = 0

    for g in ghostStates:
        d = manhattanDistance(pos, g.getPosition())
        if g.scaredTimer > 0:
            totalScared += g.scaredTimer
            scaredMin = d if scaredMin is None else min(scaredMin, d)
        else:
            activeMin = d if activeMin is None else min(activeMin, d)

    if activeMin is not None:
        if activeMin <= 1:
            score -= 2000.0
        elif activeMin == 2:
            score -= 800.0
        elif activeMin == 3:
            score -= 200.0
        else:
            score -= 2.0 / (activeMin + 1.0)

    if scaredMin is not None:
        score += 35.0 / (scaredMin + 1.0)
        score += 0.2 * totalScared

    try:
        if Directions.STOP in currentGameState.getLegalPacmanActions():
            score -= 1.5
    except Exception:
        pass

    return score

better = betterEvaluationFunction