

import json
from pathlib import Path
import numpy as np

from config.rootPath import getRootPath
from config.simulationConfig.simulationConfigFields import SimulationConfigFields



class SimulationManager():

    def __init__(self, fileName: str) -> None:
        self.fileName = fileName
        self.filePath = self._getFilePath()
        self.values = self._loadConfig()

    def getIMax(self) -> int:
        return self._getValue(SimulationConfigFields.I_MAX)  
    
    def getJayMax(self) -> int:
        return self._getValue(SimulationConfigFields.J_MAX)
    
    def getXLength(self) -> int:
        return self._getValue(SimulationConfigFields.X_LENGTH)
    
    def getYLength(self) -> int:
        return self._getValue(SimulationConfigFields.Y_LENGTH)
    
    def getDeltaTime(self) -> float:
        return self._getValue(SimulationConfigFields.DELTA_TIME)
    
    def getEndTime(self) -> float:
        return self._getValue(SimulationConfigFields.END_TIME)
    
    def getTau(self) -> float:
        return self._getValue(SimulationConfigFields.TAU)
    
    def getEpsilon(self) -> float:
        return self._getValue(SimulationConfigFields.EPSILON)
    
    def getOmega(self) -> float:
        return self._getValue(SimulationConfigFields.OMEGA)
    
    def getAlpha(self) -> float:
        return self._getValue(SimulationConfigFields.ALPHA)
    
    def getReynoldsNumber(self) -> float:
        return self._getValue(SimulationConfigFields.REYNOLDS_NUBMER)
    
    def getMaxItterations(self) -> int:
        return self._getValue(SimulationConfigFields.MAX_ITTERATIONS)
    
    def getVArray(self) -> np.array:
        return np.zeros((self.getJayMax(), self.getIMax()))
    
    def getUArray(self) -> np.array:
        return np.zeros((self.getJayMax(), self.getIMax()))
    
    def getPressureArray(self) -> np.array:
        return np.zeros((self.getJayMax(), self.getIMax()))

    def _getValue(self, simulationConfigField: SimulationConfigFields) -> any:
        return self.values.get(simulationConfigField.value)
        
    def getFileName(self) -> str:
        return self.fileName

    def _getFilePath(self) -> Path:
        rootPath = getRootPath()
        return rootPath.joinpath("config", "simulationConfig", f"{self.fileName}.json")
    
    def _loadConfig(self) -> dict:
        with open(self.filePath, "r") as file:
            return json.load(file)