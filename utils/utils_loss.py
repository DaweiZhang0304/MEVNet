import torch.nn as nn
import torch
import torch.fft as fft



class WassLoss(nn.Module):
    def __init__(self, spatial_freq_weight=1):
        super().__init__()
        self.spatial_freq_weight = spatial_freq_weight

    def wasserstein_distance(self, data, target_data):
        data_sorted = torch.sort(data.flatten())[0]
        target_sorted = torch.sort(target_data.flatten())[0]

        return torch.sum(torch.abs(data_sorted - target_sorted)) / torch.numel(
            data_sorted
        )

    def frequency_domain_transform(self, x):
        """Compute the FFT of the input tensor and return the magnitude."""
        fft_magnitude = torch.abs(fft.fft2(x))
        return fft_magnitude

    def wass_loss(self, translated, gt, sigma_map):
        # Compute the noise for each channel
        noise = translated - gt
        # Generate Gaussian target distribution using the global mean and standard deviation
        # sigma_map  = torch.sigmoid(F.instance_norm(sigma_map))
        
        gaussian_target = torch.randn_like(noise) * sigma_map * 50

        # Initialize total losses
        total_gaussian_loss = 0.0
        total_rayleigh_loss = 0.0

        # Compute the loss for each channel
        for c in range(translated.shape[1]):
            # Get the noise and Gaussian target for the current channel
            channel_noise = noise[:, c, :, :]
            channel_gaussian_target = gaussian_target[:, c, :, :]

            # Wasserstein distance for Gaussian distribution (channel-wise)
            gaussian_loss = self.wasserstein_distance(
                channel_noise, channel_gaussian_target
            )

            # Rayleigh distribution in the frequency domain (channel-wise)
            freq_noise = self.frequency_domain_transform(channel_noise)
            rayleigh_target = self.frequency_domain_transform(channel_gaussian_target)

            # Wasserstein distance for Rayleigh distribution (channel-wise)
            rayleigh_loss = self.wasserstein_distance(freq_noise, rayleigh_target)

            # Accumulate the losses for each channel
            total_gaussian_loss += gaussian_loss
            total_rayleigh_loss += rayleigh_loss

        # Combine the losses for all channels
        total_loss = (
            total_gaussian_loss + total_rayleigh_loss * self.spatial_freq_weight
        )

        return total_loss

    def forward(self, translated, target, sigma_map):
        loss = self.wass_loss(translated, target, sigma_map)
        return loss.mean()  # Average over all channels

