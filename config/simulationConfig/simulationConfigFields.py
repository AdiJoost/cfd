from enum import Enum

class SimulationConfigFields(Enum):
    I_MAX = "iMax"
    J_MAX = "jMax"
    X_LENGTH = "xLength"
    Y_LENGTH = "yLength"
    DELTA_TIME = "delta_time"
    END_TIME = "end_time"
    TAU = "tau"
    EPSILON = "epsilon"
    OMEGA = "omega"
    ALPHA = "alpha"
    REYNOLDS_NUBMER = "reynoldsNumber"
    MAX_ITTERATIONS = "iterMax"
    