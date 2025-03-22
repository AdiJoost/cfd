# ProjectTemplate
This is a project template for a python project. It features a logging, data handling, config handling and a template to build and run it in docker. The main idea behind the template is to have a clean working tree.
To enable the setup, adjust the rootFolder in the ./config/rootPath.py file to your root folder name.

## Environment Variables
The template has a file for loading environment variables into the docker container. It is recommended to run the command, to prevent pushing the variables to the git repo.
### Ignore Environment Variables Change
        git update-index --assume-unchanged dockerEnvironmentVariables.env

## Create the docker file
In the Dockerfile, set the workdir to your rootFolder name

### Create the docker image
    docker build -t <your_app_name> .
### Run the container with the environments variables
    docker run --env-file=dockerEnvironmentVariables.env <your_app_name>
