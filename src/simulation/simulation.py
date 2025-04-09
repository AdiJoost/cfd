import matplotlib.pyplot as plt
import numpy as np

from config.simulationConfig.simulationManager import SimulationManager
from src.solvers.idkTheName import step


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
            self._computeFStar(deltaTime)
            self._computeGStar(deltaTime)
            self._updatePressure(deltaTime)
            self._updateU(deltaTime)
            self._updateV(deltaTime)
            self.time += deltaTime
        self.plot()

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
        self.uArray[:, 0] = 2 - self.uArray[:, 1]
        self.uArray[:, self.jMax - 1] = 1 - self.uArray[:, self.jMax - 2]
        self.uArray[0,:] = 0
        self.uArray[self.iMax - 1,:] = 0
        self.vArray[0,:] = 2 - self.vArray[1,:]
        self.vArray[self.iMax - 1,:] = 2 - self.vArray[self.iMax - 2,:]
        self.vArray[:, self.jMax - 1] = 0
        self.vArray[:, 0] = 0

    def _computeFStar(self, deltaTime) -> None:
        d2U_dX2 = self._get_d2U_dX2()
        d2U_DY2 = self._get_d2U_dY2()
        dU2_dX = self._get_dU2_dX()
        dUV_dY = self._get_dUV_dy()
        self.fStar = self.uArray + deltaTime * ((1/ self.reynoldsNumber) *(d2U_dX2 + d2U_DY2) - dU2_dX - dUV_dY)

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
        result[1:-1, :] = (1/self.xLength) * ( ((self.uArray[1:-1,:] + self.uArray[2:,:]) / 2)**2 - ((self.uArray[1:-1,:] + self.uArray[:-2,:]) / 2)**2 ) 
        alphaParam = np.zeros_like(self.uArray)
        alphaParam[1:-1, :] = (self.alpha / self.xLength) * ((np.abs(self.uArray[1:-1,:] + self.uArray[2:,:]) * (self.uArray[1:-1,:] - self.uArray[2:,:]) / 4 ) - (np.abs(self.uArray[1:-1,:] + self.uArray[:-2,:]) * (self.uArray[:-2,:] - self.uArray[1:-1,:]) / 4 ))
        return result + alphaParam

    def _computeGStar(self, deltaTime) -> None:
        d2V_dY2 = self._get_d2V_dY2()
        d2V_dX2 = self._get_d2V_dX2()
        dV2_dY = self._get_dV2_dY()
        dUV_dX = self._get_dUV_dX()
        self.gStar = self.vArray + deltaTime * ((1/ self.reynoldsNumber) *(d2V_dX2 + d2V_dY2) - dV2_dY - dUV_dX)

    def _get_d2V_dX2(self):
        result = np.zeros_like(self.vArray)
        result[1:-1, :] = (self.vArray[2:,:] - 2*self.vArray[1:-1, :] + self.vArray[:-2, :]) /(self.xLength**2)
        return result

    def _get_d2V_dY2(self):
        result = np.zeros_like(self.vArray)
        result[:, 1:-1] = (self.vArray[:,2:] - 2*self.vArray[:, 1:-1] + self.vArray[:, :-2]) /(self.yLength**2)
        return result
    
    def _get_dV2_dY(self):
        result = np.zeros_like(self.vArray)
        result[:, 1:-1] = (1/self.yLength) * ( ((self.vArray[:,1:-1] + self.vArray[:,2:]) / 2)**2 - ((self.vArray[:,1:-1] + self.vArray[:,:-2]) / 2)**2 ) 
        alphaParam = np.zeros_like(self.vArray)
        alphaParam[:, 1:-1] = (self.alpha / self.yLength) * ((np.abs(self.vArray[:,1:-1] + self.vArray[:,2:]) * (self.vArray[:,1:-1] - self.vArray[:,2:]) / 4 ) - (np.abs(self.vArray[:,1:-1] + self.vArray[:,:-2]) * (self.vArray[:,:-2] - self.vArray[:,1:-1]) / 4 ))
        return result + alphaParam
    
    def _get_dUV_dX(self):
        result = np.zeros_like(self.vArray)
        result[:, 1:-1] = (1/self.yLength) * ( ((self.vArray[:,1:-1] + self.vArray[:,2:]) / 2)**2 - ((self.vArray[:,1:-1] + self.vArray[:,:-2]) / 2)**2 ) 
        alphaParam = np.zeros_like(self.vArray)
        alphaParam[:, 1:-1] = (self.alpha / self.yLength) * ((np.abs(self.vArray[:,1:-1] + self.vArray[:,2:]) * (self.vArray[:,1:-1] - self.vArray[:,2:]) / 4 ) - (np.abs(self.vArray[:,1:-1] + self.vArray[:,:-2]) * (self.vArray[:,:-2] - self.vArray[:,1:-1]) / 4 ))
        return result + alphaParam

    def _updatePressure(self, dt) -> None:
        res = float("inf")
        itteration = 0
        while res > self.epsilon and itteration < 100:
            step(self.PressureArray, self.fStar, self.gStar, self.xLength, self.yLength, dt)
            itteration += 1
        self.PressureArray[0,:] = self.PressureArray[1,:]
        self.PressureArray[self.iMax - 1, :] = self.PressureArray[self.iMax - 2, :]
        self.PressureArray[:, 0] = self.PressureArray[:, 1]
        self.PressureArray[:, self.jMax - 1] = self.PressureArray[:, self.jMax - 2]


    def _updateU(self, dt) -> None:
        self.uArray[1:,:] = self.fStar[1:,:] - (dt/self.xLength)* (self.PressureArray[1:,:] - self.PressureArray[:-1,:])

    def _updateV(self, dt) -> None:
        self.vArray[:,1:] = self.gStar[:,1:] - (dt/self.yLength)* (self.PressureArray[:,1:] - self.PressureArray[:,:-1])

    def plot(self) -> None:
        fig = plt.figure(figsize=(11, 7), dpi=100)
        x = np.linspace(0, self.iMax)
        y = np.linspace(0, self.jMax)
        X, Y = np.meshgrid(x, y)
        plt.contourf(X, Y, self.PressureArray, alpha=0.5, cmap="viridis")
        plt.colorbar()
        plt.contour(X, Y, self.PressureArray, cmap="viridis")
        plt.quiver(X[::2, ::2], Y[::2, ::2], self.uArray[::2, ::2], self.vArray[::2, ::2])
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()