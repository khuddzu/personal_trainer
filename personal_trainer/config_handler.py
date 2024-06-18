import configparser
import torch
from pathlib import Path


class ConfigHandler:
    def __init__(self, config_file):
        """
        ConfigHandler class sets up all specified values set by the user to 
        train an ANI model. The ini file used in this class allows for flexible model architecture and prototyping.

        Arguments:
        config_file: path to editor.ini model configuration file; str
        """
        self.config = configparser.ConfigParser(allow_no_value=True, inline_comment_prefixes="#")
        self.config.read(config_file)
        self.global_vars = self.config['Global']
        self.trainer_vars = self.config['Trainer']
        
    def get_device(self):
        """
        This function grabs the device if present in the user's editor.ini file. 
        Most likely this function will be deleted and device will be assigned within the user's code, 
        as doing otherwise has caused bugs in the past. 
        """
        return eval(self.global_vars.get('device'))

    def get_value(self, section, key):
        """
        The get_value function is a helper function to the get_all_values function below. 
        This function takes each section within the editor.ini configuration file and sorts through
        all keys listed in each, returning each key value in it's proper typing. 

        Arguments-

        section: iterative output of self.config.sections(); Currently, in the template editor.ini file 
        the sections are 'Global' and 'Trainer'
        
        key: iterative output of self.config[section]; each key and its value is set by the user in the 
        editor.ini configuration file. The list of keys is user defined and can be extensive, therefore 
        it is not defined here. Each key in the template editor.ini serves a purpose in the current version
        of personal_trainer. Not defining a certain key value will lead to default settings. 

        """
        value = self.config[section].get(key)
        try:                                                #Check for all other types that are non-str  
            return eval(value)
        except:                                             #Return str value
            return value

    def get_all_values(self):
        config_dict = {}
        for section in self.config.sections():
            config_dict[section] = {}
            for key in self.config[section]:
                config_dict[section][key] = self.get_value(section, key)
        return config_dict
