import random
import numpy as np

from config.simulationConfig.simulationManager import SimulationManager


class Simulation():

    def __init__(self, configName: str) -> None:
        self.simulationManager = SimulationManager(configName)
        self.iMax= self.simulationManager.getIMax()
        self.jMax = self.simulationManager.getJayMax()
        self.xLength = self.simulationManager.getXLength()
        self.yLength = self.simulationManager.getYLength()
        self.maxItteration = self.simulationManager.getMaxItterations()
        self.reynoldsNumber = self.simulationManager.getReynoldsNumber()
        self.epsilon = self.simulationManager.getEpsilon()
        self.alpha = self.simulationManager.getAlpha()
        self.tau = self.simulationManager.getTau()
        self.vArray = self.simulationManager.getVArray()
        self.uArray: np.array = self.simulationManager.getUArray()
        self.PressureArray = self.simulationManager.getPressureArray()
        self.time = 0.0
        self.endTime = self.simulationManager.getEndTime()
        self.deltaTime = self.simulationManager.getDeltaTime()
        self.fStar = self.simulationManager.getVArray()
        self.gStar = self.simulationManager.getVArray()


    def run(self) -> None:
        while (self.time < self.endTime):
            print(f"Time: {self.time}")
            deltaTime = self._caluclateDeltaTime()
            self._setBoundaries()
            self._computeFStar()
            self._computeGStar()
            self._updatePressure()
            self._updateU()
            self._updateV()
            self.time += deltaTime
        print(self.uArray)
        print(self.fStar)

    def _caluclateDeltaTime(self) -> float:
        maxFromU = np.max(self.uArray)
        maxFromV = np.max(self.vArray)
        maxFromU = np.abs(maxFromU) if maxFromU != 0 else 1
        maxFromV = np.abs(maxFromV) if maxFromV != 0 else 1
        deltaTX = self.xLength / maxFromU - 0.001
        deltaTY = self.yLength / maxFromV - 0.001

        reynoldsStability = (self.reynoldsNumber / 2) * 1/(1/(self.xLength ** 2) + 1/(self.yLength ** 2))
        return self.tau * min (reynoldsStability, deltaTX, deltaTY)

    def _setBoundaries(self) -> None:
        self.uArray[0, :] = 2 - self.uArray[1, :]

    def _computeFStar(self) -> None:
        d2U_dX2 = self._get_d2U_dX2()
        d2U_DY2 = self._get_d2U_dY2()
        dU2_dX = self._get_dU2_dX()
        self.fStar = (1/ self.reynoldsNumber) *(d2U_dX2 + d2U_DY2) - dU2_dX

    def _computeGStar(self) -> None:
        pass

    def _get_d2U_dX2(self):
        result = np.zeros_like(self.uArray)
        result[1:-1, :] = (self.uArray[2:,:] - 2*self.uArray[1:-1, :] + self.uArray[:-2, :]) /(self.xLength**2)
        return result
    
    def _get_d2U_dY2(self):
        result = np.zeros_like(self.uArray)
        result[:, 1:-1] = (self.uArray[:,2:] - 2*self.uArray[:, 1:-1] + self.uArray[:, :-2]) /(self.yLength**2)
        return result
    
    def _get_dU2_dX(self):
        result = np.zeros_like(self.uArray)
        result[1:-1, :] = (1/self.xLength) * ( ((self.uArray[1:-1,:] + self.uArray[2:,:]) / 2)**2 - ((self.uArray[1:-1,:] + self.uArray[:-2,:]) / 2)**2 ) 
        alphaParam = np.zeros_like(self.uArray)
        alphaParam[1:-1, :] = (self.alpha / self.xLength) * ((np.abs(self.uArray[1:-1,:] + self.uArray[2:,:]) * (self.uArray[1:-1,:] - self.uArray[2:,:]) / 4 ) - (np.abs(self.uArray[1:-1,:] + self.uArray[:-2,:]) * (self.uArray[:-2,:] - self.uArray[1:-1,:]) / 4 ))
        return result + alphaParam
    
    def _get_dUV_dy(self):
        result = np.zeros_like(self.uArray)
        result[1:-1, :] = (1/self.xLength) * ( ((self.uArray[1:-1,:] + self.uArray[2:,:]) / 2)**2 - ((self.uArray[1:-1,:] + self.uArray[:2,:]) / 2)**2 ) 
        alphaParam = np.zeros_like(self.uArray)
        alphaParam[1:-1, :] = (self.alpha / self.xLength) * ((np.abs(self.uArray[1:-1,:] + self.uArray[2:,:]) * (self.uArray[1:-1,:] - self.uArray[2:,:]) / 4 ) - (np.abs(self.uArray[1:-1,:] + self.uArray[:-2,:]) * (self.uArray[:-2,:] - self.uArray[1:-1,:]) / 4 ))
        return result + alphaParam

    def _updatePressure(self) -> None:
        res = float("inf")
        itteration = 0
        while res > self.epsilon and itteration < self.maxItteration:
            itteration += 1

    def _updateU(self) -> None:
        pass

    def _updateV(self) -> None:
        pass