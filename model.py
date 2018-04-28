import torch
import torch.nn as nn

from modules import LateralBlock, DownSamplingBlock, UpSamplingBlock

class GridNet(nn.Module):
    def __init__(self, n_ch_init, n_ch_final,
                 n_row = 3, n_col = 6, n_chs = [32, 64, 96]):
        super().__init__()

        assert n_row == len(n_chs), 'should give num channels for each row (scale stream)'
        self.n_row = n_row
        self.n_col = n_col
        self.n_chs = n_chs

        self.lateral_init = LateralBlock(n_ch_init, n_chs[0])
        self.lateral_blocks = [[LateralBlock(n_ch, n_ch) for _ in range(n_col-1)] for n_ch in n_chs]
        self.down_blocks = [[DownSamplingBlock(ch_in, ch_out) \
                             for _ in range(int(n_col/2))]
                            for ch_in, ch_out in zip(n_chs[:-1], n_chs[1:])]
        self.up_blocks = [[UpSamplingBlock(ch_in, ch_out) \
                           for _ in range(int(n_col/2))]
                          for ch_in, ch_out in zip(n_chs[1:], n_chs[:-1])]
        self.lateral_final = LateralBlock(n_chs[0], n_ch_final)
                                    
    def forward(self, x):

        outputs = [[0 for _ in range(self.n_col)] for _ in range(self.n_row)]
        outputs[0][0] = self.lateral_init(x)
        
        for col in range(self.n_col):
            # downsampling
            if col < int(self.n_col/2):
                for row in range(self.n_row-1):
                    outputs[row+1][col] += self.down_blocks[row][col](outputs[row][col])
            # upsamlping
            else:
                # row : descending order
                for row in reversed(range(self.n_row-1)):
                    outputs[row][col] += self.up_blocks[row][col-int(self.n_col/2)](outputs[row+1][col])
            if col < self.n_col-1:
                for row in range(self.n_row):
                    outputs[row][col+1] = self.lateral_blocks[row][col](outputs[row][col])

        return self.lateral_final(outputs[0][-1])
