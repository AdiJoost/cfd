import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from config.simulationConfig.simulationManager import SimulationManager
from src.ghia.ghia import plotGhis
from src.solvers.idkTheName import step, stepWithIndices
from src.vtkMapper.vtkMapper import VTKMapper

  
class Simulation():

    def __init__(self, configName: str) -> None:
        self.simulationManager = SimulationManager(configName)
        self.iMax= self.simulationManager.getIMax()
        self.jMax = self.simulationManager.getJayMax()
        self.xLength = self.simulationManager.getXLength()
        self.yLength = self.simulationManager.getYLength()
        self.maxItteration = self.simulationManager.getMaxItterations()
        self.reynoldsNumber:float = self.simulationManager.getReynoldsNumber()
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

        self.j = slice(1, -1)
        self.i = slice(1, -1)
        self.jPuls1 = slice(2, self.jMax)
        self.iPlus1 = slice(2, self.iMax)
        self.jMinus1 = slice(0, self.jMax - 2)
        self.iMinus1 = slice(0, self.iMax - 2)


    def run(self) -> None:
        pbar = tqdm(total=self.endTime)
        itterations = 0
        while (self.time < self.endTime):
            deltaTime = self._caluclateDeltaTime()
            self._setBoundaries()
            #if itterations %1000 == 0:
            #    self.plot()
            itterations += 1
            self._computeFStar(deltaTime)
            self._computeGStar(deltaTime)
            self._setBoundaries()
            self._updatePressure(deltaTime)
            self._updateU(deltaTime)
            self._updateV(deltaTime)
            self.time += deltaTime
            pbar.update(deltaTime)
            self._checkErrorValues()
            
        pbar.close()
        self.plot()
        plotGhis(self.uArray, self.vArray)
        self.save()

    def save(self) -> None:
        VTKMapper(fileName="lid_driven_cavity", uVector=self.uArray, vVector=self.vArray, pressureVector=self.PressureArray).export()

    def _caluclateDeltaTime(self) -> float:
        maxFromU = np.max(self.uArray)
        maxFromV = np.max(self.vArray)
        maxFromU = np.abs(maxFromU) if maxFromU != 0 else 1
        maxFromV = np.abs(maxFromV) if maxFromV != 0 else 1
        deltaTX = self.xLength / maxFromU
        deltaTY = self.yLength / maxFromV

        reynoldsStability = (self.reynoldsNumber / 2) * (( (1/ (self.xLength**2)) + (1 / (self.yLength**2)) ) **-1)
        time = self.tau * min (reynoldsStability, deltaTX, deltaTY, 0.05)
        return time if time > 0.0 else 0.00001

    def _setBoundaries(self) -> None:
        self.uArray[:, 0] = 0
        self.uArray[:, self.iMax - 2] = 0
        self.uArray[0,:] = - self.uArray[1,:]
        self.uArray[self.jMax - 1, :] = 2 - self.uArray[self.jMax - 2,:]

        self.vArray[0,:] = 0
        self.vArray[self.jMax - 2,:] = 0
        self.vArray[:, self.iMax - 1] = - self.vArray[:, self.iMax - 2]
        self.vArray[:, 0] = - self.vArray[:, 1]

    def _computeFStar(self, deltaTime) -> None:
        d2U_dX2 = self._get_d2U_dX2()
        d2U_DY2 = self._get_d2U_dY2()
        dU2_dX = self._get_dU2_dX()
        dUV_dY = self._get_dUV_dy()
        self.fStar = self.uArray + deltaTime * ((( 1.0/self.reynoldsNumber) *(d2U_dX2 + d2U_DY2)) - dU2_dX - dUV_dY)
        self.fStar[:,0] = self.uArray[:,0]
        self.fStar[:, self.iMax - 1] = self.uArray[:,self.iMax - 1]
        self.fStar[:, self.iMax - 2] = self.uArray[:,self.iMax - 2]  
    

    def _get_d2U_dX2(self):
        result = np.zeros_like(self.uArray)
        result[:, self.i] = (self.uArray[:,self.iMinus1] - 2*self.uArray[:, self.i] + self.uArray[:, self.iPlus1]) /(self.xLength**2)
        return result
    
    def _get_d2U_dY2(self):
        result = np.zeros_like(self.uArray)
        result[self.j, :] = (self.uArray[self.jPuls1,:] - 2*self.uArray[self.j,:] + self.uArray[self.jMinus1, :]) /(self.yLength**2)
        return result
    
    def _get_dU2_dX(self):
        result = np.zeros_like(self.uArray)
        result[:, self.i] = (1/self.xLength) * ( ((self.uArray[:,self.i] + self.uArray[:,self.iPlus1]) / 2)**2 - ((self.uArray[:,self.iMinus1] + self.uArray[:,self.i]) / 2)**2 ) 
        alphaParam = np.zeros_like(self.uArray)
        alphaParam[:, self.i] = (self.alpha / self.xLength) * ((np.abs(self.uArray[:,self.i] + self.uArray[:,self.iPlus1]) / 2 * (self.uArray[:,self.i] - self.uArray[:,self.iPlus1]) / 2 ) - (np.abs(self.uArray[:,self.iMinus1] + self.uArray[:,self.i]) / 2 * (self.uArray[:,self.iMinus1] - self.uArray[:,self.i]) / 2 ))
        return (1-self.alpha) * result + alphaParam
    
    def _get_dUV_dy(self):
        result = np.zeros_like(self.uArray)
        result[self.j,self.i] = (1/self.yLength) * ((self.vArray[self.j, self.i] + self.vArray[self.j, self.iPlus1]) * (self.uArray[self.j, self.i] + self.uArray[self.jPuls1, self.i]) / 4)
        result[self.j, self.i] -= (1/self.yLength) * (self.vArray[self.jMinus1, self.i] + self.vArray[self.jMinus1, self.iPlus1]) *(self.uArray[self.jMinus1, self.i] + self.uArray[self.j, self.i]) / 4 
        alphaParam = np.zeros_like(self.uArray)
        alphaParam[self.j,self.i] = (self.alpha / self.yLength) * (np.abs(self.vArray[self.j, self.i] + self.vArray[self.j, self.iPlus1])*(self.uArray[self.j,self.i] - self.uArray[self.jPuls1, self.i])) / 4
        alphaParam[self.j, self.i] -= (self.alpha / self.yLength) * (np.abs(self.vArray[self.jMinus1, self.i] + self.vArray[self.jMinus1, self.iPlus1])*(self.uArray[self.jMinus1,self.i] - self.uArray[self.j, self.i])) / 4
        return (1-self.alpha) * result + alphaParam

    def _computeGStar(self, deltaTime) -> None:
        d2V_dY2 = self._get_d2V_dY2()
        d2V_dX2 = self._get_d2V_dX2()
        dV2_dY = self._get_dV2_dY()
        dUV_dX = self._get_dUV_dX()
        self.gStar = self.vArray + deltaTime * (((1.0/self.reynoldsNumber) *(d2V_dX2 + d2V_dY2)) - dV2_dY - dUV_dX)
        self.gStar[0,:] = self.vArray[0,:]
        self.gStar[self.jMax - 1, :] = self.vArray[self.jMax - 1,:]
        self.gStar[self.jMax - 2, :] = self.vArray[self.jMax - 2,:]

    def _get_d2V_dX2(self):
        result = np.zeros_like(self.vArray)
        result[:,self.i] = (self.vArray[:,self.iPlus1] - 2*self.vArray[:,self.i] + self.vArray[:, self.iMinus1]) /(self.xLength**2)
        return result

    def _get_d2V_dY2(self):
        result = np.zeros_like(self.vArray)
        result[self.j,:] = (self.vArray[self.jPuls1,:] - 2*self.vArray[self.j, :] + self.vArray[self.jMinus1, :]) /(self.yLength**2)
        return result
    
    def _get_dV2_dY(self):
        result = np.zeros_like(self.vArray)
        result[self.j, :] = (1/self.yLength) * ( ((self.vArray[self.j, :] + self.vArray[self.jPuls1,:]) / 2)**2 - (((self.vArray[self.jMinus1,:] + self.vArray[self.j,:]) / 2)**2) ) 
        alphaParam = np.zeros_like(self.vArray)
        alphaParam[self.j,:] = (self.alpha / self.yLength) * ((np.abs(self.vArray[self.j,:] + self.vArray[self.jPuls1,:]) * (self.vArray[self.j,:] - self.vArray[self.jPuls1,:]) / 4 ) - (np.abs(self.vArray[self.jMinus1,:] + self.vArray[self.j,:]) * (self.vArray[self.jMinus1,:] - self.vArray[self.j,:]) / 4 ))
        return (1-self.alpha) * result + alphaParam
    
    def _get_dUV_dX(self):
        result = np.zeros_like(self.vArray)
        result[self.j, self.i] = (1/self.xLength) * ( (self.uArray[self.j, self.i] + self.uArray[self.jPuls1, self.i]) / 2 * (self.vArray[self.j, self.i]+ self.vArray[self.j,self.iPlus1]) / 2)
        result[self.j, self.i] -= (1/self.xLength) * ( (self.uArray[self.j, self.iMinus1] + self.uArray[self.jPuls1, self.iMinus1]) / 2 * (self.vArray[self.j, self.iMinus1]+ self.vArray[self.j,self.i]) / 2)
        alphaParam = np.zeros_like(self.vArray)
        alphaParam[self.j, self.i] = (self.alpha / self.xLength) * (np.abs(self.uArray[self.j, self.i] + self.uArray[self.jPuls1,self.i]) / 2) * ((self.vArray[self.j,self.i] - self.vArray[self.j,self.iPlus1]) / 2) - (np.abs(self.uArray[self.j, self.iMinus1] + self.uArray[self.jPuls1,self.iMinus1]) / 2) * ((self.vArray[self.j, self.iMinus1] - self.vArray[self.j,self.i]) / 2)
        return (1-self.alpha) * result + alphaParam

    def _updatePressure(self, dt) -> None:
        res = float("inf")
        itteration = 0
        self.PressureArray[:,0] = self.PressureArray[:,1]
        self.PressureArray[0,:] = self.PressureArray[1,:]
        self.PressureArray[:, self.iMax -1] = self.PressureArray[:, self.iMax - 2]
        self.PressureArray[self.jMax - 1, :] = self.PressureArray[self.jMax - 2, :]
        xOld = self.PressureArray.copy()
        while res > self.epsilon and itteration < 10000:
            stepWithIndices(self.PressureArray, self.fStar, self.gStar, self.xLength, self.yLength, dt, self.j, self.jMinus1, self.jPuls1, self.i, self.iMinus1, self.iPlus1)
            self.PressureArray[0,:] = self.PressureArray[1,:]
            self.PressureArray[self.jMax - 1, :] = self.PressureArray[self.jMax - 2, :]
            self.PressureArray[:, 0] = self.PressureArray[:, 1]
            self.PressureArray[:, self.iMax - 1] = self.PressureArray[:, self.iMax - 2]
            diff = self.PressureArray - xOld
            res = np.max(np.abs(diff))
            xOld = self.PressureArray.copy()
            itteration += 1


    def _updateU(self, dt) -> None:
        self.uArray[self.j,self.i] = self.fStar[self.j,self.i] - (dt/self.xLength)* (self.PressureArray[self.j,self.iPlus1] - self.PressureArray[self.j,self.i])

    def _updateV(self, dt) -> None:
        self.vArray[self.j, self.i] = self.gStar[self.j,self.i] - (dt/self.yLength)* (self.PressureArray[self.jPuls1,self.i] - self.PressureArray[self.j,self.i])

    def _checkErrorValues(self) -> None:
        if (np.any(np.isnan(self.uArray)) or np.any((np.isinf(self.uArray)))):
            self.plot()
            raise ValueError("A element is NaN or Inf in uArray")
        if (np.any(np.isnan(self.vArray)) or np.any((np.isinf(self.vArray)))):
            self.plot()
            raise ValueError("A element is NaN or Inf in vArray")
        if (np.any(np.isnan(self.PressureArray)) or np.any((np.isinf(self.PressureArray)))):
            self.plot()
            raise ValueError("A element is NaN or Inf in PressureArray")

    def plot(self) -> None:
        fig = plt.figure(figsize=(11, 7), dpi=100)
        u_padded = np.pad(self.uArray, pad_width=1, mode='constant', constant_values=0)
        v_padded = np.pad(self.vArray, pad_width=1, mode='constant', constant_values=0)
        p_padded = np.pad(self.PressureArray, pad_width=1, mode='constant', constant_values=0)
        ny, nx = self.PressureArray.shape
        x = np.linspace(0, self.iMax)
        y = np.linspace(0, self.jMax)
        x = np.linspace(0, self.iMax, nx + 2)
        y = np.linspace(0, self.jMax, ny + 2)
        X, Y = np.meshgrid(x, y)
        plt.contourf(X, Y, p_padded, alpha=0.5, cmap="viridis")
        plt.colorbar()
        plt.contour(X, Y, p_padded, cmap="viridis")
        plt.quiver(X[::2, ::2], Y[::2, ::2], u_padded[::2, ::2], v_padded[::2, ::2])
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.gca().get_xaxis().set_visible(False)
        plt.gca().get_yaxis().set_visible(False)
        plt.show()