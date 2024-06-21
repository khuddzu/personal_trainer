import configparser


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

    def get_value(self, section, key):
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

        Notes
        -----
        Activation in global section might be depricated.
        """
        value = self.config[section].get(key)
        try:
            return eval(value)  # Try to evaluate the value
        except BaseException:
            return value  # Return as string if evaluation fails

    def get_all_values(self):
        """
        Retrieve all configuration values, organized by section.

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
                config_dict[section][key] = self.get_value(section, key)
        return config_dict
