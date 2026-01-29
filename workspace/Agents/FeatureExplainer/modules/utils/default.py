import yaml
from pathlib import Path
import math

BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIRS = BASE_DIR / "dirs.yaml"

class Autosort_Dirs():
    base_folder = 'Autosort_data'
    def __init__(self, processed_folder):
        self.autosort_folder = Path(processed_folder)/self.base_folder
        self.autosort_folder.mkdir(exist_ok=True)
        with open(CONFIG_DIRS, "r") as f:
            self.config = yaml.safe_load(f)
        self._convert_dict_to_attr(self.config)
    
    def update(self,**kwargs):
        for key,value in kwargs.items():
            if key not in self.config:
                raise KeyError(f'Wrong parameter key input: {key}={value}')
        self._convert_dict_to_attr(kwargs)
        
    def unpack(self):
        filtered_dict = {k: v for k, v in self.__dict__.items() if k not in  ["config"]}
        for i, (k,v) in enumerate(filtered_dict.items()):
            print(f'{k}: {v}')
            
    
    def _convert_dict_to_attr(self,config):
        for key, value in config.items():
            setattr(self, key, Path(self.autosort_folder/value))
    
    def __repr__(self):
        filtered_dict = {k: v for k, v in self.__dict__.items() if k not in ["config"]}
        return f"{filtered_dict}"
    
class Autosort_Params():
    def __init__(self, dirs):
        self.params_path = dirs.params_path
        with open(self.params_path, "r") as f:
            self.config = yaml.safe_load(f)
        self._convert_dict_to_attr(self.config)
    
    def update(self,**kwargs):
        for key,value in kwargs.items():
            if key not in self.config:
                raise KeyError(f'Wrong parameter key input: {key}={value}')
        self._convert_dict_to_attr(kwargs)
        with open(self.params_path, 'w') as file:
            updated_params = {k:getattr(self,k) for k in self.config.keys()}
            yaml.dump(updated_params, file, default_flow_style=False)
        
    def unpack(self):
        filtered_dict = {k: v for k, v in self.__dict__.items() if k not in  ["config"]}
        for i, (k,v) in enumerate(filtered_dict.items()):
            print(f'{k}: {v}')
            
    
    def _convert_dict_to_attr(self,config):
        for key, value in config.items():
            setattr(self, key, value)
    
    def __repr__(self):
        filtered_dict = {k: v for k, v in self.__dict__.items() if k not in ["config"]}
        return f"{filtered_dict}"