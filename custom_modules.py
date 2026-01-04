import torch
import torch.nn as nn


class VoV_GSCSP(nn.Module):
    def __init__(self, c1, c2, n=1, g=1, e=0.5, k_size=7):
        super().__init__()
        c_ = int(c2 * e)

        self.lk_conv = nn.Sequential(
            nn.Conv2d(c1, c_, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(),
            nn.Conv2d(c_, c_, kernel_size=k_size, stride=1,
                      padding=k_size // 2, groups=g, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU()
        )

        self.split_conv = nn.Conv2d(c_, c_ // 2, kernel_size=1, bias=False)

        self.dw_conv = nn.Sequential(
            nn.Conv2d(c_ // 2, c_ // 2, kernel_size=3, stride=1,
                      padding=1, groups=c_ // 2, bias=False),
            nn.BatchNorm2d(c_ // 2),
            nn.SiLU(),
            nn.Conv2d(c_ // 2, c_ // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_ // 2),
            nn.SiLU()
        )

        self.final_conv = nn.Conv2d(c_, c2, kernel_size=1, bias=False)

    def forward(self, x):
        x_main = self.lk_conv(x)
        split_1, split_2 = torch.split(x_main, x_main.size(1) // 2, dim=1)
        branch = self.dw_conv(split_1)
        out = torch.cat([branch, split_2], dim=1)
        return self.final_conv(out)