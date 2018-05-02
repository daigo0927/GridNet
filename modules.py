import torch
import torch.nn as nn

class LateralBlock(nn.Module):

    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.f = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size = 3, padding = 1),
            nn.PReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size = 3, padding = 1)
        )
        if ch_in != ch_out:
            self.conv = nn.Conv2d(ch_in, ch_out, kernel_size = 3, padding = 1)

    def forward(self, x):
        fx = self.f(x)
        if fx.shape[1] != x.shape[1]:
            x = self.conv(x)
            
        return fx + x

    
class DownSamplingBlock(nn.Module):

    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.f = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size = 3, stride = 2, padding = 1),
            nn.PReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size = 3, padding = 1)
        )

    def forward(self, x):
        return self.f(x)

class UpSamplingBlock(nn.Module):

    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.f = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = False),
            # nn.UpsamplingNearest2d(scale_factor = 2),
            nn.PReLU(),
            nn.Conv2d(ch_in, ch_out, kernel_size = 3, padding = 1),
            nn.PReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size = 3, padding = 1)
        )

    def forward(self, x):
        return self.f(x)
