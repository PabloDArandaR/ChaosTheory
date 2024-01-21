from abc import abstractmethod

class model:

    @abstractmethod
    def __init__(self, n_dim):
        self.n_dim = n_dim
        pass
    
    @abstractmethod
    def process_step(self):
        pass

    @abstractmethod
    def get_state():
        pass