import numpy as np
import sys

def add1(x): # analogous to lateral block
    return x+1

def devide2(x): # to downsampling 
    return x/2

def times2(x): # to upsampling
    return x*2

class ToyGrid(object):

    def __init__(self, n_row, n_col):
        self.n_row, self.n_col = n_row, n_col
        
        self.add_blocks = [[add1 for _ in range(n_col-1)] for _ in range(n_row)]
        self.devide_blocks = [[devide2 for _ in range(int(n_col/2))]\
                             for _ in range(n_row-1)]
        self.times_blocks = [[times2 for _ in range(int(n_col-1))]\
                             for _ in range(n_row-1)]

    def forward(self):
        status = np.zeros((self.n_row, self.n_col))
        status[0,0] = 1

        for col in range(self.n_col):
            # downsampling
            if col < int(self.n_col/2):
                for row in range(self.n_row-1):
                    status[row+1][col] += self.devide_blocks[row][col](status[row, col])
            # upsampling
            else:
                for row in reversed(range(self.n_row-1)):
                    status[row, col] += self.times_blocks[row][col-int(self.n_col/2)](status[row+1, col])

            if col < self.n_col-1:
                for row in range(self.n_row):
                    status[row, col+1] = self.add_blocks[row][col](status[row, col])

        return status

if __name__ == '__main__':
    args = [3, 6] # default n_row and n_col
    assert len(sys.argv) <= 3, 'too much (>2) argument has passed'
    for i in range(1, len(sys.argv)):
        args[i-1] = int(sys.argv[i])
    print(f'Build toy-grid with {args[0]}rows, {args[1]}cols')
    
    toy = ToyGrid(*args)
    print(toy.forward())
