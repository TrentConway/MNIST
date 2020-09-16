#!/usr/bin/env python3

import yaml

class Parameters():
    def __init__(self, file):
        self.paramsFile = file
        self.params = self.__loadParams()

    def __loadParams(self):
        if not self.paramsFile.endswith('.yml'):
            raise TypeError("Parameters was supplied with , when it requires a yaml file".format(self.paramsFile))
        with open(self.paramsFile) as file:
            return yaml.load(file, Loader=yaml.FullLoader)

    def getModelParams(self):
        return self.params['model']
