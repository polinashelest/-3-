import util
#import mdp
from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):


    def __init__(self, mdp, discount=0.9, iterations=100):

        super().__init__()
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # Счетчик - это дикт со значением по умолчанию 0

        for i in range(self.iterations):
            for state in self.mdp.getStates():
                possibleActions = self.mdp.getPossibleActions(state)
                valuesForActions = util.Counter()
                for action in possibleActions:
                    transitionStatesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)
                    valueState = 0
                    for transition in transitionStatesAndProbs:
                        valueState += transition[1] * (
                                    self.mdp.getReward(state, action, transition[0]) + self.discount * self.values[
                                transition[0]])
                    valuesForActions[action] = valueState
                self.values[state] = valuesForActions[valuesForActions.argMax()]

    def getValue(self, state):
        """
          Возвращает значение состояния (вычисляется в __init__).
        """
        return self.values[state]

    def getQValue(self, state, action):
        """
          Q- пары действий состояния
           (после указанного количества итераций значения
           проходит).
        """
        transitionStatesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)
        qValue = 0
        for transition in transitionStatesAndProbs:
            qValue += transition[1] * self.mdp.getReward(state, action, transition[0]) + self.values[transition[0]]

        return qValue

    def getPolicy(self, state):
        """
          Политика является лучшим действием в данном состоянии
           в соответствии со значениями, вычисленными путем итерации значений.
           Можно разорвать отношения любым удобным для вас способом. Обратите внимание, что если
           отсутствуют какие-либо  действия, как это имеет место в
           состояние терминала,  должно вернуть None.
        """
        if self.mdp.isTerminal(state):
            return None

        possibleActions = self.mdp.getPossibleActions(state)

        valuesForActions = util.Counter()
        for action in possibleActions:
            transitionStatesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)
            valueState = 0
            for transition in transitionStatesAndProbs:
                valueState += transition[1] * (
                            self.mdp.getReward(state, action, transition[0]) + self.discount * self.values[
                        transition[0]])
            valuesForActions[action] = valueState

        if valuesForActions.totalCount() == 0:
            import random
            return possibleActions[int(random.random() * len(possibleActions))]
        else:
            valueToReturn = valuesForActions.argMax()

            return valueToReturn

    def getAction(self, state):
        return self.getPolicy(state)