import torch
import torch.nn as nn

# ===================================================================
# This file contains the models used in this project: DnCNN, ResCNN, and FFTResCNN
# ===================================================================


class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers):
        super(DnCNN, self).__init__()
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=3, padding=1, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out


class ResCNN(nn.Module):
    def __init__(self, channels, num_of_layers, num_of_resblocks):
        super(ResCNN, self).__init__()
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))
        num_of_main = num_of_layers - 2 - num_of_resblocks * 2
        num_of_before = num_of_main // 2
        num_of_after = num_of_main - num_of_before
        for _ in range(num_of_before):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_resblocks):
            layers.append(ResBlock(features))
            layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_after):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=3, padding=1, bias=False))
        self.rescnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.rescnn(x)
        return out


class ResBlock(nn.Module):
    def __init__(self, features, norm='backward'):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features)
        )
        self.dim = features
        self.norm = norm

    def forward(self, x):
        return self.main(x) + x


class FFTResCNN(nn.Module):
    def __init__(self, channels, num_of_layers, num_of_resblocks):
        super(FFTResCNN, self).__init__()
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))
        num_of_main = num_of_layers - 2 - num_of_resblocks*2
        num_of_before = num_of_main // 2
        num_of_after = num_of_main - num_of_before
        for _ in range(num_of_before):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_resblocks):
            layers.append(FFT_ResBlock(features))
            layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_after):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=3, padding=1, bias=False))
        self.fft_rescnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.fft_rescnn(x)
        return out


class FFT_ResBlock(nn.Module):
    def __init__(self, features, norm='backward'):
        super(FFT_ResBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features)
        )
        self.main_fft = nn.Sequential(
            nn.Conv2d(features * 2, features * 2, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(features * 2, features * 2, kernel_size=3, padding=1, bias=False)
        )
        self.dim = features
        self.norm = norm

    def forward(self, x):
        _, _, H, W = x.shape
        dim = 1
        fft = torch.fft.rfft2(x, norm=self.norm)
        fft_real = fft.real
        fft_imag = fft.imag
        fft_complex = torch.cat([fft_real, fft_imag], dim=dim)
        fft = self.main_fft(fft_complex)
        fft_real, fft_imag = torch.chunk(fft, 2, dim=dim)
        fft = torch.complex(fft_real, fft_imag)
        fft = torch.fft.irfft2(fft, s=(H, W), norm=self.norm)
        return self.main(x) + x + fft
