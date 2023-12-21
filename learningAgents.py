from game import Agent

import util

class ValueEstimationAgent(Agent):
  def __init__(self, alpha=1.0, epsilon=0.05, gamma=0.8, numTraining=10):
    super().__init__()
    self.alpha = float(alpha)
    self.epsilon = float(epsilon)
    self.discount = float(gamma)
    self.numTraining = int(numTraining)

  def getQValue(self, state, action):
    util.raiseNotDefined()

  def getValue(self, state):

    util.raiseNotDefined()

  def getPolicy(self, state):
    util.raiseNotDefined()

  def getAction(self, state):

    util.raiseNotDefined()