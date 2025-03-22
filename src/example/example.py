from config.configManager import loadConfig
import os

def main():
    print("Hello "+ loadConfig())
    print(os.environ.get("APP_ENV"))


    

