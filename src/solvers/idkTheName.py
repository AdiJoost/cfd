import numpy as np

def step(pArray, fStar, gStar, dx, dy, dt):
    pd = pArray.copy()
    b = np.zeros_like(gStar)
    b[1:,:] = dt*(fStar[1:,:] - fStar[:-1,:])
    b[:,1:] = b[:,1:] + dt*(gStar[:,1:] - gStar[:,:-1]) 
    pArray[1:-1,1:-1] = (((pd[1:-1, 2:] + pd[1:-1, :-2]) * dy**2 +
                    (pd[2:, 1:-1] + pd[:-2, 1:-1]) * dx**2 -
                    b[1:-1, 1:-1] * dx**2 * dy**2) / 
                    (2 * (dx**2 + dy**2)))