import numpy as np

def step(pArray, fStar, gStar, dx, dy, dt):
    pd = pArray.copy()
    b = np.zeros_like(gStar)
    b[:,1:] = dt*(fStar[:,1:] - fStar[:,:-1])
    b[1:,:] = b[1:,:] + dt*(gStar[1:,:] - gStar[:-1,:]) 
    pArray[1:-1,1:-1] = (((pd[2:,1:-1] + pd[:-2,1:-1]) * dy**2 +
                    (pd[1:-1,2:] + pd[1:-1,:-2]) * dx**2 -
                    b[1:-1, 1:-1] * dx**2 * dy**2) / 
                    (2 * (dx**2 + dy**2)))
    
def stepWithIndices(pArray, fStar, gStar, dx, dy, dt, j, jMinus1, jPlus1, i, iMinus1, iPlus1):
    pArray[j,i] = ((
                        (1.0/dt
                            ) * (
                                ((fStar[j,i] - fStar[j, iMinus1]) * (1.0/dx)
                                 ) + (
                                 (gStar[j,i] - gStar[jMinus1, i]) * (1.0/dy)        
                                 )     
                            ) * (dy**2) * (dx**2)
                        ) - (
                        dy**2 * (pArray[j, iPlus1] + pArray[j, iMinus1])
                        ) - (
                        dx**2 * (pArray[jPlus1, i] + pArray[jMinus1, i]))
                    ) * (
                    1.0/-4.0)