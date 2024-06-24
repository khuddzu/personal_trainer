import configparser
import torchani
import os
import importlib

class ConfigHandler:
    def __init__(self, config_file):
        """
        Initialize the ConfigHandler class.

        Parameters
        ----------
        config_file : str
            Path to the editor.ini model configuration file.
        """
        self.config = configparser.ConfigParser(
            allow_no_value=True, inline_comment_prefixes="#"
        )
        self.config.read(config_file)
        self.global_vars = self.config["Global"]
        self.trainer_vars = self.config["Trainer"]

    def get_device(self):
        """
        Retrieve the device specified in the configuration file.

        Returns
        -------
        torch.device
            The device specified in the configuration file.

        Notes
        -----
        This method might be removed in the future, and the device should be assigned within the user's code
        to avoid potential bugs.
        """
        return eval(self.global_vars.get("device"))

    def _convert(self, section, key):
        """
        Retrieve the value of a key in a given section and convert it to the appropriate type.

        Parameters
        ----------
        section : str
            The section in the configuration file.
        key : str
            The key within the section.

        Returns
        -------
        any
            The value of the key, converted to the appropriate type.

        """
        value = self.config[section].get(key)
        try:
            return eval(value)  # Try to evaluate the value
        except BaseException:
            return value  # Return as string if evaluation fails
    
    def load_config(self):
        """
        Retrieve all configuration values, organized by section.
        AEV param files and elements are also established based on the preference of like1x or like2x. 

        Returns
        -------
        dict
            A dictionary containing all configuration data, with sections as keys and
            key-value pairs as values.

        """
        config_dict = {}
        for section in self.config.sections():
            config_dict[section] = {}
            for key in self.config[section]:
                config_dict[section][key] = self._convert(section, key)
        
        torchani_path = importlib.util.find_spec('torchani').submodule_search_locations[0]

        global_config = config_dict['Global']

        assert not (global_config['netlike1x'] and global_config['netlike2x']), "In configuration file, netlike1x and netlike2x cannot both be True"

        if global_config['netlike1x']:
            global_config['constants'] = os.path.join(torchani_path, 'resources', 'ani-1x_8x', 'rHCNO-5.2R_16-3.5A_a4-8.params')
            global_config['elements'] = ['H', 'C', 'N', 'O']
        elif global_config['netlike2x']:
            global_config['constants'] = os.path.join(torchani_path, 'resources', 'ani-2x_8x', 'rHCNOSFCl-5.1R_16-3.5A_a8-4.params')
            global_config['elements'] = ['H', 'C', 'N', 'O', 'S', 'F', 'Cl']
        else:
            assert 'constants' in global_config and global_config['constants'] is not None, "Constants pathway  not specified in configuration file."
            assert 'elements' in global_config and global_config['elements'] is not None, "Elements not specified in configuration file."

        return config_dict
