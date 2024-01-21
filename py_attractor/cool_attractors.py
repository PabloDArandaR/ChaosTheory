import py_attractor.model as model
import numpy as np

class aizawaAttractor (model.model):
    def __init__(self, initial_state: np.array, dt: float, a: float, b: float, c: float, d: float, e: float, f: float):
        self.n_dim = 3
        self.x = initial_state[0]
        self.y = initial_state[1]
        self.z = initial_state[2]
        self.derivative_x = 0
        self.derivative_y = 0
        self.derivative_z = 0
        self.dt = dt
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f
    
    def calculate_derivative(self):
        dx = (self.z - self.b)*self.x - self.d*self.y
        dy = self.d*self.x + (self.z - self.b)*self.y
        dz = self.c + self.a*self.z - (self.z**3)/3 - (self.x**2 + self.y**2)*(1 + self.e*self.z) + self.f*self.z*(self.x**3)

        self.derivative_x, self.derivative_y, self.derivative_z = dx, dy, dz
    
    def process_step(self):
        self.calculate_derivative()

        self.x += self.derivative_x*self.dt
        self.y += self.derivative_y*self.dt
        self.z += self.derivative_z*self.dt
    
    def get_state(self):
        return self.x, self.y, self.z


class chenLeeAttractor(model.model):
    def __init__(self, initial_state: np.array, dt: float, a: float, b: float, d: float):
        self.n_dim = 3
        self.x = initial_state[0]
        self.y = initial_state[1]
        self.z = initial_state[2]
        self.derivative_x = 0
        self.derivative_y = 0
        self.derivative_z = 0
        self.dt = dt
        self.a = a
        self.b = b
        self.d = d
    
    def calculate_derivative(self):
        dx = self.a*self.x - self.y*self.z
        dy = self.b*self.y + self.x*self.z
        dz = self.d*self.z + self.x*self.y/3

        self.derivative_x, self.derivative_y, self.derivative_z = dx, dy, dz
    
    def process_step(self):
        self.calculate_derivative()

        self.x += self.derivative_x*self.dt
        self.y += self.derivative_y*self.dt
        self.z += self.derivative_z*self.dt
    
    def get_state(self):
        return self.x, self.y, self.z


class rosslerAttractor(model.model):
    def __init__(self, initial_state: np.array, dt: float, a: float, b: float, c: float):
        self.n_dim = 3
        self.x = initial_state[0]
        self.y = initial_state[1]
        self.z = initial_state[2]
        self.derivative_x = 0
        self.derivative_y = 0
        self.derivative_z = 0
        self.dt = dt
        self.a = a
        self.b = b
        self.c = c
    
    def calculate_derivative(self):
        dx = - self.y - self.z
        dy = self.x + self.a*self.y
        dz = self.b + self.z*(self.x - self.c)

        self.derivative_x, self.derivative_y, self.derivative_z = dx, dy, dz
    
    def process_step(self):
        self.calculate_derivative()

        self.x += self.derivative_x*self.dt
        self.y += self.derivative_y*self.dt
        self.z += self.derivative_z*self.dt
    
    def get_state(self):
        return self.x, self.y, self.z


class arneodoAttractor(model.model):
    def __init__(self, initial_state: np.array, dt: float, a: float, b: float, c: float):
        self.n_dim = 3
        self.x = initial_state[0]
        self.y = initial_state[1]
        self.z = initial_state[2]
        self.derivative_x = 0
        self.derivative_y = 0
        self.derivative_z = 0
        self.dt = dt
        self.a = a
        self.b = b
        self.c = c
    
    def calculate_derivative(self):
        dx = self.y
        dy = self.z
        dz = -self.a*self.x - self.b*self.y - self.z + self.c*self.x**3

        self.derivative_x, self.derivative_y, self.derivative_z = dx, dy, dz
    
    def process_step(self):
        self.calculate_derivative()

        self.x += self.derivative_x*self.dt
        self.y += self.derivative_y*self.dt
        self.z += self.derivative_z*self.dt
    
    def get_state(self):
        return self.x, self.y, self.z


class sprottBAttractor(model.model):
    def __init__(self, initial_state: np.array, dt: float, a: float, b: float, c: float):
        self.n_dim = 3
        self.x = initial_state[0]
        self.y = initial_state[1]
        self.z = initial_state[2]
        self.derivative_x = 0
        self.derivative_y = 0
        self.derivative_z = 0
        self.dt = dt
        self.a = a
        self.b = b
        self.c = c
    
    def calculate_derivative(self):
        dx = self.a*self.y*self.z
        dy = self.x - self.b*self.y
        dz = self.c - self.x*self.y

        self.derivative_x, self.derivative_y, self.derivative_z = dx, dy, dz
    
    def process_step(self):
        self.calculate_derivative()

        self.x += self.derivative_x*self.dt
        self.y += self.derivative_y*self.dt
        self.z += self.derivative_z*self.dt
    
    def get_state(self):
        return self.x, self.y, self.z


class sprottLinzFAttractor(model.model):
    def __init__(self, initial_state: np.array, dt: float, a: float):
        self.n_dim = 3
        self.x = initial_state[0]
        self.y = initial_state[1]
        self.z = initial_state[2]
        self.derivative_x = 0
        self.derivative_y = 0
        self.derivative_z = 0
        self.dt = dt
        self.a = a
    
    def calculate_derivative(self):
        dx = self.y + self.z
        dy = -self.x + self.a*self.y
        dz = self.x**2 - self.z

        self.derivative_x, self.derivative_y, self.derivative_z = dx, dy, dz
    
    def process_step(self):
        self.calculate_derivative()

        self.x += self.derivative_x*self.dt
        self.y += self.derivative_y*self.dt
        self.z += self.derivative_z*self.dt
    
    def get_state(self):
        return self.x, self.y, self.z


class dadrasAttractor(model.model):
    def __init__(self, initial_state: np.array, dt: float, p: float, o: float, r: float, c: float, e: float):
        self.n_dim = 3
        self.x = initial_state[0]
        self.y = initial_state[1]
        self.z = initial_state[2]
        self.derivative_x = 0
        self.derivative_y = 0
        self.derivative_z = 0
        self.dt = dt
        self.p = p
        self.o = o
        self.r = r
        self.c = c
        self.e = e
    
    def calculate_derivative(self):
        dx = self.y - self.p*self.x + self.o*self.y*self.z
        dy = self.r*self.y - self.x*self.z + self.z
        dz = self.c*self.x*self.y - self.e*self.z

        self.derivative_x, self.derivative_y, self.derivative_z = dx, dy, dz
    
    def process_step(self):
        self.calculate_derivative()

        self.x += self.derivative_x*self.dt
        self.y += self.derivative_y*self.dt
        self.z += self.derivative_z*self.dt
    
    def get_state(self):
        return self.x, self.y, self.z


class halvorsenAttractor(model.model):
    def __init__(self, initial_state: np.array, dt: float, a: float):
        self.n_dim = 3
        self.x = initial_state[0]
        self.y = initial_state[1]
        self.z = initial_state[2]
        self.derivative_x = 0
        self.derivative_y = 0
        self.derivative_z = 0
        self.dt = dt
        self.a = a
    
    def calculate_derivative(self):
        dx = - self.a*self.x - 4*self.y - 4*self.z - self.y*self.y
        dy = - self.a*self.y - 4*self.z - 4*self.x - self.z*self.z
        dz = - self.a*self.z - 4*self.x - 4*self.y - self.x*self.x

        self.derivative_x, self.derivative_y, self.derivative_z = dx, dy, dz
    
    def process_step(self):
        self.calculate_derivative()

        self.x += self.derivative_x*self.dt
        self.y += self.derivative_y*self.dt
        self.z += self.derivative_z*self.dt
    
    def get_state(self):
        return self.x, self.y, self.z
