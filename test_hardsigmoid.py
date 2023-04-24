import torch
from torch import nn


class HardSigmoid(nn.Module):
    def __init__(self, slope: float = 1 / 6, offset: float = 0.5, left_interval: float = -3.0, right_interval: float = 3.0):
        super(HardSigmoid, self).__init__()
        self.slope = slope
        self.offset = offset
        self.left_interval = left_interval
        self.right_interval = right_interval
    '''forward'''

    def forward(self, x):
        left_mask = (x <= self.left_interval)
        right_mask = (x >= self.right_interval)
        otherwise = (x > self.left_interval) & (x < self.right_interval)
        output = x.clone()
        output[left_mask] = 0.0
        output[right_mask] = 1.0
        output[otherwise] = x[otherwise] * self.slope + self.offset
        return output

if __name__ == "__main__":
    x = torch.tensor([-3.15,1.234,0.8834,3.21,4.54,2.47],dtype=torch.float32)
    h1 = HardSigmoid(slope=0.2)
    h2 = nn.Hardsigmoid()
    r1 = h1(x)
    r2 = h2(x)
    print("r1:{}".format(r1))
    print("r2:{}".format(r2))