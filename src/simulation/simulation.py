import numpy

from config.simulationConfig.simulationManager import SimulationManager


class Simulation():

    def __init__(self, configName: str) -> None:
        self.simulationManager = SimulationManager(configName)
        self.iMax= self.simulationManager.getIMax()


    def run(self) -> None:
        print(self.iMax)