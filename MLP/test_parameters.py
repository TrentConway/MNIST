import pytest as pytest
from parameters import Parameters
import os 
import yaml


# Global Test Variables
data_containsNoModel = {'sport' : ['soccer', 'football', 'basketball', 'cricket', 'hockey', 'table tennis'], 'countries' : ['Pakistan', 'USA', 'India', 'China', 'Germany', 'France', 'Spain']} 
data_containsModel = {'model' : ['soccer', 'football', 'basketball', 'cricket', 'hockey', 'table tennis'], 'countries' : ['Pakistan', 'USA', 'India', 'China', 'Germany', 'France', 'Spain']}


# Test Helper functions
def createFile(data, fileName):
    with open(fileName, 'w') as file:
        yaml.dump(data, file)
    file.close()

def deleteFile(fileName):
    if os.path.exists(fileName):
        os.remove(fileName)


# Invalid Parameters file 
def test_notYamlFile():
    fileName = "notyaml.json"
    open(fileName, "w+").close()
    with pytest.raises(TypeError):
        params = Parameters(fileName)
    deleteFile(fileName)

# File does not contain model 
def test_containsNoModel():
    fileName = "test_noModel.yml"
    createFile(data_containsNoModel, fileName) 
    with pytest.raises(KeyError):
        params = Parameters(fileName).getModelParams()
    deleteFile(fileName)

# getModel only returns the model parameters
def test_onylReturnsModelParameters():
    fileName = "test_noModel.yml"    
    createFile(data_containsModel, fileName) 
    params = Parameters(fileName)
    assert params.getModelParams(), data_containsModel["model"] 
    deleteFile(fileName)
