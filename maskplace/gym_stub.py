"""Minimal gym stub for Mode A (training-free).
Only provides the base class and spaces that place_env.py inherits.
Use ONLY if gym==0.21.0 fails to install."""

class Env:
    """Minimal gym.Env stub."""
    pass

class Space:
    def __init__(self, shape=None, dtype=None):
        self.shape = shape
        self.dtype = dtype

class Box(Space):
    def __init__(self, low=0, high=1, shape=None, dtype=None):
        super().__init__(shape, dtype)
        self.low = low
        self.high = high

class Discrete(Space):
    def __init__(self, n):
        self.n = int(n)
        super().__init__(shape=(), dtype=int)

    def contains(self, x):
        try:
            x = int(x)
        except (TypeError, ValueError):
            return False
        return 0 <= x < self.n

class spaces:
    Box = Box
    Discrete = Discrete
