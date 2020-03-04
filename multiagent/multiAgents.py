import util
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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

    def evaluationFunction(self, currentGameState, action):
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
        # Dist metric to Food
        foodList = util.matrixAsList(newFood.data) # Locations of all food
        totDistFood = 1 # init to 1 to prevent div by 0
        closestFoodDist = 999999
        furthestFoodDist = 1
        for food in foodList:
            dist = util.manhattanDistance(newPos, food)
            totDistFood += dist
            if dist < closestFoodDist:
                closestFoodDist = dist
            if dist > furthestFoodDist:
                furthestFoodDist = dist

        # Use of "total food distance" and/or "furthest distance" leads to a tie in many cases and thus leaves 
        # pacman in STOP action and away from ghost.

        # Distance metric to ghosts
        totGhostDist = 1 # init to 1 to prevent div by 0
        tooCloseGhost = 0
        scaredGhost = 10
        for ghostIdx in range(len(newGhostStates)):
            ghostState = newGhostStates[ghostIdx]
            scaredTime = newScaredTimes[ghostIdx]
            if scaredTime > 0: # If ghost in deactive state, don't care about it's distance
                scaredGhost += 50
                continue
            else:
                ghostDist = util.manhattanDistance(newPos, ghostState.configuration.pos)
                totGhostDist += ghostDist
                if ghostDist <= 1:
                    tooCloseGhost -= 200 # negative number

        #totGhostDist = 1 # init to 1 to prevent div by 0
        #for ghostPos in successorGameState.getGhostPositions():
        #    ghostDist = util.manhattanDistance(newPos, ghostPos)
        #    totGhostDist += ghostDist

        # Adding this will prevent pacman from stalling, 
        # By making it oscillate. So not useful. Might as well stall
        #stopCost = 0
        #if action == 'Stop':
        #    stopCost = 20
         
        #effScore = - stopCost + successorGameState.getScore() + 1/float(totDistFood) + 1/float(closestFoodDist) + 1/float(furthestFoodDist) + totGhostDist + extraCostGhost + scaredGhost
        #effScore = successorGameState.getScore() + 1/float(totDistFood) + 1/float(closestFoodDist) + 1/float(furthestFoodDist) + totGhostDist + extraCostGhost + scaredGhost
        #effScore = successorGameState.getScore() + 1/float(furthestFoodDist) + totGhostDist + extraCostGhost + scaredGhost
        #effScore = successorGameState.getScore() + (1/float(furthestFoodDist)) - (1/float(totGhostDist)) #+ extraCostGhost + scaredGhost
        effScore = successorGameState.getScore() + (1/float(closestFoodDist)) - (1/float(totGhostDist)) + tooCloseGhost + scaredGhost

        return effScore


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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def minimax(self, depth, agent, gameState):
        
        if agent >= gameState.getNumAgents():
            agent = 0
            depth += 1

        if (depth==self.depth or gameState.isWin() or gameState.isLose()):
            return self.evaluationFunction(gameState)
        # Agent and depth are independent factors. Not directly correlated.
        #agent = depth % gameState.getNumAgents() # 0 = pacman = root, 1 = ghost1, 2 = ghost2 ... so on

        # Pacman Related MAX actions
        elif(agent == 0):
            pacValues = []
            possActions = gameState.getLegalActions(agent)
                
            for action in possActions:
                newGameState = gameState.generateSuccessor(agent, action)
                #pacValues.append( minimax(currState, depth+1, agent+1) )
                pacValues.append( self.minimax(depth, agent+1, newGameState) )

            maxValue = max(pacValues)
            #maxPosition = [pacValues == maxValue] # This syntax works only for numpy arrays
            #if(len(maxPosition) > 1): # If multiple max locations exist
            #    maxPosition = maxPosition[0]

            maxPosition = [i for i, x in enumerate(pacValues) if x == maxValue]
            maxAction = possActions[maxPosition[0]]
            return maxAction                

        # Ghost related MIN actions
        else:
            ghostValues = []
            possActions = gameState.getLegalActions(agent)
                
            for action in possActions:
                newGameState = gameState.generateSuccessor(agent, action)
                #ghostValues.append( minimax(currState, depth+1, agent+1) )
                ghostValues.append( self.minimax(depth, agent+1, newGameState) )

            minValue = min(ghostValues)
            #minPosition = [ghostValues == minValue] # This syntax only works for numpy arrays
            #if(len(minPosition) > 1): # If multiple min locations exist
            #    minPosition = minPosition[0]

            minPosition = [i for i, x in enumerate(ghostValues) if x == minValue]
            minAction = possActions[minPosition[0]]
            return minAction         


    def getAction(self, gameState):
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
        """
        "*** YOUR CODE HERE ***"
        #finalDepth = self.depth
        #currDepth = 0

        #action = self.minimax(currDepth, gameState)
        action = self.minimax(0, 0, gameState)
        return action



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction