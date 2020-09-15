import yaml

class Parameters():
    def __init__(self, file):
        self.paramsFile = file
        self.params = self.__loadParams()

    def __loadParams(self):
        with open(self.paramsFile) as file:
            return yaml.load(file, Loader=yaml.FullLoader)

    def getModelParams(self):
        return self.params['model']
