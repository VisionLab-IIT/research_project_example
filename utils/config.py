import yaml


class Config():
    def __init__(self):
        self.config_dict = None

    def load(self, config_path):
        """
        Loading dictionary from YAML file
        """
        with open(config_path, 'r') as f:
            self.config_dict = yaml.safe_load(f)

    def update(self, args):
        """
        Updating config values with command line arguments
        """
        args_dict = vars(args)

        for k, v in args_dict.items():
            if (k in self.config_dict.keys()) and (v is not None):
                self.config_dict[k] = v

    def save(self, save_path):
        """
        Saving config to a YAML file
        """
        with open(save_path, "w") as f:
            yaml.safe_dump(self.config_dict, f)

    def __getitem__(self, key):
        return self.config_dict[key]
    
    def __str__(self):
        """
        Formatting for printing
        """
        # Creating YAML-like string
        config_str = yaml.safe_dump(self.config_dict)
        # Removing unnecessary newline character at the end
        config_str = config_str.rstrip("\n")
        
        return config_str