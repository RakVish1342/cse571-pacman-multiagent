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

    # Taking the overall max/min after collecting all leaf node values works, 
    # BUT is not good for alpha-beta pruning later on. So it is better to update max/min
    # values on the fly, for each new node explored.
    #def minimax(self, depth, agent, gameState):
        
    #    finalDepth = self.depth
    #    totAgents = gameState.getNumAgents()

    #    if agent >= totAgents:
    #        agent = 0 # Finished going through one set of max-min-min-min layers, so reset the agent to pacman
    #        depth += 1 # Increment depth once one set is finished.

    #    if (depth==finalDepth or gameState.isWin() or gameState.isLose()):
    #        return ("Dummy Action", self.evaluationFunction(gameState)) # Dummy value added since minimax is passing about action-value pairs. 
    #    # Agent and depth are independent factors. Not directly correlated.
    #    #agent = depth % gameState.getNumAgents() # 0 = pacman = root, 1 = ghost1, 2 = ghost2 ... so on

    #    else: 
    #        # Pacman Related MAX actions
    #        if(agent == 0):
    #            pacActionValues = []
    #            possActions = gameState.getLegalActions(agent)
                
    #            for action in possActions:
    #                newGameState = gameState.generateSuccessor(agent, action)
    #                #pacValues.append( minimax(currState, depth+1, agent+1) )
    #                pacActionValues.append( self.minimax(depth, agent+1, newGameState) )
    #            pacValues = [val for act, val in pacActionValues]
    #            maxValue = max(pacValues)
    #            #maxPosition = [pacValues == maxValue] # This syntax works only for numpy arrays
    #            #if(len(maxPosition) > 1): # If multiple max locations exist
    #            #    maxPosition = maxPosition[0]

    #            maxPosition = [i for i, x in enumerate(pacValues) if x == maxValue]
    #            maxAction = possActions[maxPosition[0]]
    #            return (maxAction, maxValue)

    #        # Ghost related MIN actions
    #        else:
    #            ghostActionValues = []
    #            possActions = gameState.getLegalActions(agent)
                
    #            for action in possActions:
    #                newGameState = gameState.generateSuccessor(agent, action)
    #                #ghostValues.append( minimax(currState, depth+1, agent+1) )
    #                ghostActionValues.append( self.minimax(depth, agent+1, newGameState) )
    #            ghostValues = [val for act, val in ghostActionValues]
    #            minValue = min(ghostValues)
    #            #minPosition = [ghostValues == minValue] # This syntax only works for numpy arrays
    #            #if(len(minPosition) > 1): # If multiple min locations exist
    #            #    minPosition = minPosition[0]

    #            minPosition = [i for i, x in enumerate(ghostValues) if x == minValue]
    #            minAction = possActions[minPosition[0]]
    #            return (minAction, minValue)


    # Implemented the on the fly max setting
    def minimax(self, depth, agent, gameState):
        
        finalDepth = self.depth
        totAgents = gameState.getNumAgents()

        if agent >= totAgents:
            agent = 0 # Finished going through one set of max-min-min-min layers, so reset the agent to pacman
            depth += 1 # Increment depth once one set is finished.

        if (depth==finalDepth or gameState.isWin() or gameState.isLose()):
            return ("Dummy Action", self.evaluationFunction(gameState)) # Dummy value added since minimax is passing about action-value pairs. 
        # Agent and depth are independent factors. Not directly correlated.
        #agent = depth % gameState.getNumAgents() # 0 = pacman = root, 1 = ghost1, 2 = ghost2 ... so on

        else: 
            # Pacman Related MAX actions
            
            #if(agent == 0):
            #    pacActionValues = []
            #    possActions = gameState.getLegalActions(agent)
                
            #    for action in possActions:
            #        newGameState = gameState.generateSuccessor(agent, action)
            #        #pacValues.append( minimax(currState, depth+1, agent+1) )
            #        pacActionValues.append( self.minimax(depth, agent+1, newGameState) )
            #    pacValues = [val for act, val in pacActionValues]
            #    maxValue = max(pacValues)
            #    #maxPosition = [pacValues == maxValue] # This syntax works only for numpy arrays
            #    #if(len(maxPosition) > 1): # If multiple max locations exist
            #    #    maxPosition = maxPosition[0]

            #    maxPosition = [i for i, x in enumerate(pacValues) if x == maxValue]
            #    maxAction = possActions[maxPosition[0]]
            #    return (maxAction, maxValue)

            if(agent == 0):
                possActions = gameState.getLegalActions(agent)
                maxValue = -float("Inf")
                maxAction = None
                
                for action in possActions:
                    newGameState = gameState.generateSuccessor(agent, action)
                    _, val = self.minimax(depth, agent+1, newGameState)
                    if val > maxValue:
                        maxValue = val
                        maxAction = action

                return (maxAction, maxValue)

            # Ghost related MIN actions
            else:

                #ghostActionValues = []
                #possActions = gameState.getLegalActions(agent)
                
                #for action in possActions:
                #    newGameState = gameState.generateSuccessor(agent, action)
                #    #ghostValues.append( minimax(currState, depth+1, agent+1) )
                #    ghostActionValues.append( self.minimax(depth, agent+1, newGameState) )
                #ghostValues = [val for act, val in ghostActionValues]
                #minValue = min(ghostValues)
                ##minPosition = [ghostValues == minValue] # This syntax only works for numpy arrays
                ##if(len(minPosition) > 1): # If multiple min locations exist
                ##    minPosition = minPosition[0]

                #minPosition = [i for i, x in enumerate(ghostValues) if x == minValue]
                #minAction = possActions[minPosition[0]]
                #return (minAction, minValue)

                possActions = gameState.getLegalActions(agent)
                minValue = float("Inf")
                minAction = None
                
                for action in possActions:
                    newGameState = gameState.generateSuccessor(agent, action)
                    _, val = self.minimax(depth, agent+1, newGameState)
                    if val < minValue:
                        minValue = val
                        minAction = action
                return (minAction, minValue)


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
        #action = self.minimax(0, 0, gameState)
        actionValue = self.minimax(0, 0, gameState)
        action = actionValue[0]
        return action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    #def minimax(self, depth, agent, gameState, alphabeta):
    def minimax(self, depth, agent, gameState, alpha, beta):
        
        finalDepth = self.depth
        totAgents = gameState.getNumAgents()

        if agent >= totAgents:
            agent = 0 # Finished going through one set of max-min-min-min layers, so reset the agent to pacman
            depth += 1 # Increment depth once one set is finished.

        if (depth==finalDepth or gameState.isWin() or gameState.isLose()):
            return ("Dummy Action", self.evaluationFunction(gameState)) # Dummy value added since minimax is passing about action-value pairs. 
        # Agent and depth are independent factors. Not directly correlated.
        #agent = depth % gameState.getNumAgents() # 0 = pacman = root, 1 = ghost1, 2 = ghost2 ... so on

        else: 
            # Pacman Related MAX actions
            if(agent == 0):
                possActions = gameState.getLegalActions(agent)
                maxValue = -float("Inf")
                maxAction = None
                for action in possActions:
                    newGameState = gameState.generateSuccessor(agent, action)
                    #_, val = self.minimax(depth, agent+1, newGameState, alphabeta)
                    _, val = self.minimax(depth, agent+1, newGameState, alpha, beta)
                    if val > maxValue:
                        maxValue = val
                        maxAction = action

                    # Alpha Beta pruning logic
                    # If curr value of node is larger than existing beta (best MIN value toward root)
                    #if maxValue > alphabeta[1]:
                    #    return (maxAction, maxValue)
                    ## update alpha
                    #alphabeta[0] = max(alphabeta[0], maxValue)
                    if maxValue > beta:
                        return (maxAction, maxValue)
                    # update alpha
                    alpha = max(alpha, maxValue)

                return (maxAction, maxValue)

            # Ghost related MIN actions
            else:
                possActions = gameState.getLegalActions(agent)
                minValue = float("Inf")
                minAction = None
                
                for action in possActions:
                    newGameState = gameState.generateSuccessor(agent, action)
                    #_, val = self.minimax(depth, agent+1, newGameState, alphabeta)
                    _, val = self.minimax(depth, agent+1, newGameState, alpha, beta)
                    if val < minValue:
                        minValue = val
                        minAction = action

                    # Alpha Beta pruning logic
                    # If curr value of node is smaller than existing alpha (best MAX value toward root)
                    #if minValue < alphabeta[0]:
                    #    return (minAction, minValue)
                    ## update beta
                    #alphabeta[1] = min(alphabeta[1], minValue)
                    if minValue < alpha:
                        return (minAction, minValue)
                    # update beta
                    beta = min(beta, minValue)

                return (minAction, minValue)

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #alphabeta = [-float("Inf"), float("Inf")]
        alpha = -float("Inf")
        beta = float("Inf")
        #actionValue = self.minimax(0, 0, gameState, alphabeta)
        actionValue = self.minimax(0, 0, gameState, alpha, beta)
        action = actionValue[0]
        return action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    # Implemented the on the fly max setting
    def minimax(self, depth, agent, gameState):
        
        finalDepth = self.depth
        totAgents = gameState.getNumAgents()

        if agent >= totAgents:
            agent = 0 # Finished going through one set of max-min-min-min layers, so reset the agent to pacman
            depth += 1 # Increment depth once one set is finished.

        if (depth==finalDepth or gameState.isWin() or gameState.isLose()):
            return ("Dummy Action", self.evaluationFunction(gameState)) # Dummy value added since minimax is passing about action-value pairs. 

        else: 
            # Pacman Related MAX actions
            # Pacman is still a MAX agent. It still takes the best actions possible.
            # So no averaging/expectation layer here
            if(agent == 0):
                possActions = gameState.getLegalActions(agent)
                maxValue = -float("Inf")
                maxAction = None
                
                for action in possActions:
                    newGameState = gameState.generateSuccessor(agent, action)
                    _, val = self.minimax(depth, agent+1, newGameState)
                    if val > maxValue:
                        maxValue = val
                        maxAction = action

                return (maxAction, maxValue)

            # Ghost related MIN actions
            else:
                possActions = gameState.getLegalActions(agent)
                #expectiValue = float("Inf")
                expectiValue = []
                minAction = None
                probab = 1/len(possActions)
                
                for action in possActions:
                    newGameState = gameState.generateSuccessor(agent, action)
                    _, val = self.minimax(depth, agent+1, newGameState)
                    expectiValue.append(val)

                 # Avg value = equal probability OR in loop do: expectiValue += val * probab
                expectiValue = sum(expectiValue) / len(expectiValue)
                randActionIdx = random.randint(0, len(possActions)-1)
                minAction = possActions[randActionIdx]
                return (minAction, expectiValue)


    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        # Implementing Mini-Max agent but with Min layers having average value
        # Avg value ie. equal probab of any of the actions happening.
        # NO alpha beta pruning here since there is no guarantee of what the actual
        # value will be in the expect layer...all depends on the probability and so
        # can NOT discount any one possible values entirely 
        actionValue = self.minimax(0, 0, gameState)
        action = actionValue[0]
        return action


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: Similar to the evaluation function of the Reflex Agent. The Pacman 
      tries to avoid the ghost, go towards food and has an extra penalty when its too
      close to the ghost
    """
    "*** YOUR CODE HERE ***"
    # Same as the earlier evaluator function. Just using the current state info
    # rather than state info of the succeeding state.
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

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

    effScore = currentGameState.getScore() + (1/float(closestFoodDist)) - 0.3*(1/float(totGhostDist)) + tooCloseGhost + scaredGhost

    # Removing the scoredGhost dropped the average score to 850-950. 

    return effScore


# Abbreviation
better = betterEvaluationFunction