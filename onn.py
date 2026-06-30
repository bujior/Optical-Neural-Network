import torch
import numpy as np


def detector_region(x):
    return torch.cat((
        x[:, 46 : 66, 46 : 66].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 46 : 66, 93 : 113].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 46 : 66, 140 : 160].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 85 : 105, 46 : 66].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 85 : 105, 78 : 98].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 85 : 105, 109 : 129].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 85 : 105, 140 : 160].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 125 : 145, 46 : 66].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 125 : 145, 93 : 113].mean(dim=(1, 2)).unsqueeze(-1),
        x[:, 125 : 145, 140 : 160].mean(dim=(1, 2)).unsqueeze(-1)), dim=-1)


class DiffractiveLayer(torch.nn.Module):

    def __init__(self):
        super(DiffractiveLayer, self).__init__()
        self.size = 200                         # 200 * 200 neurons in one layer
        self.distance = 0.03                    # distance bewteen two layers (3cm)
        self.ll = 0.08                          # layer length (8cm)
        self.wl = 3e8 / 0.4e12                  # wave length
        self.wn = 2 * 3.1415926 / self.wl       # wave number
        pixel_size = self.ll / self.size
        fx = np.fft.fftfreq(self.size, d=pixel_size)
        fy = np.fft.fftfreq(self.size, d=pixel_size)
        fxx, fyy = np.meshgrid(fx, fy)
        # phi (200, 200), aligned with the unshifted order returned by torch.fft.fft2
        phi = np.square(fxx) + np.square(fyy)
        # H (200, 200)
        H = np.exp(1.0j * self.wn * self.distance) * np.exp(-1.0j * self.wl * np.pi * self.distance * phi)
        # self.H (200, 200, 2)
        H = torch.view_as_real(torch.from_numpy(H.astype(np.complex64)))
        self.register_buffer("H", H)

    def forward(self, waves):
        # waves (batch, 200, 200, 2)
        waves_complex = torch.view_as_complex(waves.contiguous())
        H_complex = torch.view_as_complex(self.H.contiguous())
        k_space = H_complex * torch.fft.fft2(waves_complex)
        # angular_spectrum (batch, 200, 200, 2)
        angular_spectrum = torch.view_as_real(torch.fft.ifft2(k_space))
        return angular_spectrum


class Net(torch.nn.Module):
    """
    phase only modulation
    """
    def __init__(self, num_layers=5):

        super(Net, self).__init__()
        # self.phase (200, 200)
        self.phase = torch.nn.ParameterList([
            torch.nn.Parameter(torch.from_numpy(2 * np.pi * np.random.random(size=(200, 200)).astype('float32')))
            for _ in range(num_layers)
        ])
        self.diffractive_layers = torch.nn.ModuleList([DiffractiveLayer() for _ in range(num_layers)])
        self.last_diffractive_layer = DiffractiveLayer()

    def forward(self, x):
        # x (batch, 200, 200, 2)
        for index, layer in enumerate(self.diffractive_layers):
            temp = layer(x)
            exp_j_phase = torch.stack((torch.cos(self.phase[index]), torch.sin(self.phase[index])), dim=-1)
            x_real = temp[..., 0] * exp_j_phase[..., 0] - temp[..., 1] * exp_j_phase[..., 1]
            x_imag = temp[..., 0] * exp_j_phase[..., 1] + temp[..., 1] * exp_j_phase[..., 0]
            x = torch.stack((x_real, x_imag), dim=-1)
        x = self.last_diffractive_layer(x)
        # Detectors measure optical intensity, not field amplitude.
        intensity = x[..., 0] * x[..., 0] + x[..., 1] * x[..., 1]
        detector_power = detector_region(intensity)
        output = torch.log(detector_power + 1e-12)
        return output


if __name__ == '__main__':
    print(Net())
