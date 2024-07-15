import json

class Config:
    def __init__(self, config_file: str) -> None:
        """
        Initializes the configuration from a JSON file.

        :param config_file: The path to the JSON configuration file.
        """
        self.config_file = config_file
        self.config_data = self._load_config()

    def _load_config(self) -> dict:
        """
        Loads the configuration data from the JSON file.

        :return: Dictionary containing the configuration data.
        """
        with open(self.config_file, 'r') as file:
            return json.load(file)

    def get(self, key: str, default=None):
        """
        Gets the value of a configuration key.

        :param key: The configuration key.
        :param default: The default value if the key is not found.
        :return: The value of the configuration key.
        """
        return self.config_data.get(key, default)

    def set(self, key: str, value) -> None:
        """
        Sets the value of a configuration key.

        :param key: The configuration key.
        :param value: The value to set.
        """
        self.config_data[key] = value
        self._save_config()

    def _save_config(self) -> None:
        """
        Saves the configuration data to the JSON file.
        """
        with open(self.config_file, 'w') as file:
            json.dump(self.config_data, file, indent=4)
