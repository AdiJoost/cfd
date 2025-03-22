from src.simulation.simulation import Simulation

def main():
    simulation = Simulation("lid_driven_cavity")
    simulation.run()

if __name__ == "__main__":
    main()