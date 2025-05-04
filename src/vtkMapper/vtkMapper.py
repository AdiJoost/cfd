import numpy as np
import pyvista as pv

from config.rootPath import getRootPath

class VTKMapper():

    VTK_ENDING = "vts"

    def __init__(self, fileName:str, uVector: np.ndarray, vVector: np.ndarray, pressureVector:np.ndarray, folderPath:str="data/visualisation") -> None:
        self.fileName = fileName
        self.folderPath = folderPath
        self.uVector = uVector
        self.vVector = vVector
        self.pressureVector = pressureVector
        self.nx, self.ny = self.pressureVector.shape
        

    def export(self):
        x = np.linspace(0, 1, self.nx)
        y = np.linspace(0, 1, self.ny)
        xx, yy = np.meshgrid(x,y)
        points = np.zeros((self.nx * self.ny, 3))
        points[:, 0] = xx.ravel()
        points[:, 1] = yy.ravel()
        
        grid = pv.StructuredGrid()
        grid.points = points
        grid.dimensions = (self.nx, self.ny, 1)
        
        grid["pressure"] = self.pressureVector.ravel(order="F")
        velocities = np.stack((self.uVector, self.vVector, np.zeros_like(self.uVector)), axis=1)
        grid["velocity"] = velocities.reshape(-1, 3, order="F")
        print(grid)
        grid.save(self.getSavePath())


    def getSavePath(self) -> str:
        rootPath = getRootPath()
        path = rootPath.joinpath(self.folderPath)
        path.mkdir(parents=True, exist_ok=True)
        filepath = path.joinpath(f"{self.fileName}.{self.VTK_ENDING}")
        return str(filepath)