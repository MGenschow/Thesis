import json
import os

class ModelIdStorage():
    def __init__(self, file_path: str = 'model_ids.json') -> None:
        """Interact with the model id storage file."""

        self.file_path = file_path

        # Check if the file exists
        if os.path.exists(file_path):
            # Read the existing dictionary from the file
            with open(file_path, 'r') as file:
                self.data = json.load(file)
        else:
            # If the file doesn't exist, create an empty dictionary
            self.data = {}

        # Convert all keys to integer
        self.data = {int(k): v for k, v in self.data.items()}
    
    def new_id(self) -> int:
        """Create a new model id."""
        self.new_id = 0 if len(self.data) == 0 else max(self.data) + 1
        return self.new_id
    
    def store_info(self, values: dict) -> None:
        """Update dictionary with new key-value pair and save to file again."""

        if not hasattr(self, 'new_id'):
            raise AttributeError('First create a new id')
        
        self.data[self.new_id] = values
        with open(self.file_path, 'w') as file:
            json.dump(self.data, file)
