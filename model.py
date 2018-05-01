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
        
        self.lateral_blocks = [[None for _ in range(n_col-1)] for _ in range(n_row)]
        for r, n_ch in enumerate(n_chs):
            for c in range(n_col-1):
                setattr(self, f'lateral_{r}_{c}', LateralBlock(n_ch, n_ch))
                self.lateral_blocks[r][c] = eval(f'self.lateral_{r}_{c}')

        self.down_blocks = [[None for _ in range(int(n_col/2))] for _ in range(n_row-1)]
        for r, (ch_in, ch_out) in enumerate(zip(n_chs[:-1], n_chs[1:])):
            for c in range(int(n_col/2)):
                setattr(self, f'down_{r}_{c}', DownSamplingBlock(ch_in, ch_out))
                self.down_blocks[r][c] = eval(f'self.down_{r}_{c}')

        self.up_blocks = [[None for _ in range(int(n_col/2))] for _ in range(n_row-1)]
        for r, (ch_in, ch_out) in enumerate(zip(n_chs[1:], n_chs[:-1])):
            for c in range(int(n_col/2)):
                setattr(self, f'up_{r}_{c}', UpSamplingBlock(ch_in, ch_out))
                self.up_blocks[r][c] = eval(f'self.up_{r}_{c}')
                
        self.lateral_final = LateralBlock(n_chs[0], n_ch_final)
                                    
    def forward(self, x):

        # outputs = [[0 for _ in range(self.n_col)] for _ in range(self.n_row)]
        # outputs[0][0] = self.lateral_init(x)
        
        # for col in range(self.n_col):
        #     # downsampling
        #     if col < int(self.n_col/2):
        #         for row in range(self.n_row-1):
        #             outputs[row+1][col] += self.down_blocks[row][col](outputs[row][col])
        #     # upsamlping
        #     else:
        #         for row in reversed(range(self.n_row-1)): # row : descending order
        #             outputs[row][col] += self.up_blocks[row][col-int(self.n_col/2)](outputs[row+1][col])
        #     # lateral stream
        #     if col < self.n_col-1:
        #         for row in range(self.n_row):
        #             outputs[row][col+1] = self.lateral_blocks[row][col](outputs[row][col])

        state_00 = self.lateral_init(x)
        state_10 = self.down_0_0(state_00)
        state_20 = self.down_1_0(state_10)

        state_01 = self.lateral_0_0(state_00)
        state_11 = self.down_0_1(state_01) + self.lateral_1_0(state_10)
        state_21 = self.down_1_1(state_11) + self.lateral_2_0(state_20)

        state_02 = self.lateral_0_1(state_01)
        state_12 = self.down_0_2(state_02) + self.lateral_1_1(state_11)
        state_22 = self.down_1_2(state_12) + self.lateral_2_1(state_21)

        state_23 = self.lateral_2_2(state_22)
        state_13 = self.up_1_0(state_23) + self.lateral_1_2(state_12)
        state_03 = self.up_0_0(state_13) + self.lateral_0_2(state_02)

        state_24 = self.lateral_2_3(state_23)
        state_14 = self.up_1_1(state_24) + self.lateral_1_3(state_13)
        state_04 = self.up_0_1(state_14) + self.lateral_0_3(state_03)

        state_25 = self.lateral_2_4(state_24)
        state_15 = self.up_1_2(state_25) + self.lateral_1_4(state_14)
        state_05 = self.up_0_2(state_15) + self.lateral_0_4(state_04)

        return self.lateral_final(state_05)
        
        # return self.lateral_final(outputs[0][-1])

    # def forwardall(self, x):

    #     outputs = [[0 for _ in range(self.n_col)] for _ in range(self.n_row)]
    #     outputs[0][0] = self.lateral_init(x)
        
    #     for col in range(self.n_col):
    #         # downsampling
    #         if col < int(self.n_col/2):
    #             for row in range(self.n_row-1):
    #                 outputs[row+1][col] += self.down_blocks[row][col](outputs[row][col])
    #         # upsamlping
    #         else:
    #             for row in reversed(range(self.n_row-1)): # row : descending order
    #                 outputs[row][col] += self.up_blocks[row][col-int(self.n_col/2)](outputs[row+1][col])
    #         # lateral stream
    #         if col < self.n_col-1:
    #             for row in range(self.n_row):
    #                 outputs[row][col+1] = self.lateral_blocks[row][col](outputs[row][col])

    #     return outputs
