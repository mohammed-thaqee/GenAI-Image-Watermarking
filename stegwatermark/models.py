"""Encoder, decoder, and critic.

The shapes match models.py in the StegaStamp repo for a 400×400 image and a
100-bit message. The encoder returns a residual. The caller adds it to the
cover, which is still in [0, 1]. The decoder returns raw logits.
"""

import torch
from torch import nn
import torch.nn.functional as F


def _he(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _up_k2(x, conv):
    """Nearest ×2, then a 2×2 conv with the same padding Keras uses."""
    x = F.interpolate(x, scale_factor=2, mode="nearest")
    x = F.pad(x, (0, 1, 0, 1))
    return F.relu(conv(x))


class Encoder(nn.Module):
    def __init__(self, secret_size=100):
        super().__init__()
        self.secret_size = secret_size
        self.secret_dense = nn.Linear(secret_size, 50 * 50 * 3)

        self.conv1 = nn.Conv2d(6, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=2, padding=1)

        self.up6 = nn.Conv2d(256, 128, 2, padding=0)
        self.conv6 = nn.Conv2d(256, 128, 3, padding=1)
        self.up7 = nn.Conv2d(128, 64, 2, padding=0)
        self.conv7 = nn.Conv2d(128, 64, 3, padding=1)
        self.up8 = nn.Conv2d(64, 32, 2, padding=0)
        self.conv8 = nn.Conv2d(64, 32, 3, padding=1)
        self.up9 = nn.Conv2d(32, 32, 2, padding=0)
        self.conv9 = nn.Conv2d(32 + 32 + 6, 32, 3, padding=1)
        self.residual = nn.Conv2d(32, 3, 1, padding=0)

        self.apply(_he)
        nn.init.kaiming_normal_(self.residual.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.residual.bias)

    def forward(self, secret, image):
        # secret: [B, secret_size] in {0, 1}. image: [B, 3, 400, 400] in [0, 1].
        if image.shape[-2:] != (400, 400):
            raise ValueError("encoder expects 400×400 images, got %s" % (tuple(image.shape[-2:]),))
        secret_c = secret - 0.5
        image_c = image - 0.5

        hidden = F.relu(self.secret_dense(secret_c))
        hidden = hidden.view(-1, 3, 50, 50)
        hidden = F.interpolate(hidden, scale_factor=8, mode="nearest")

        x = torch.cat([hidden, image_c], dim=1)
        c1 = F.relu(self.conv1(x))
        c2 = F.relu(self.conv2(c1))
        c3 = F.relu(self.conv3(c2))
        c4 = F.relu(self.conv4(c3))
        c5 = F.relu(self.conv5(c4))

        u6 = _up_k2(c5, self.up6)
        u6 = F.relu(self.conv6(torch.cat([c4, u6], dim=1)))
        u7 = _up_k2(u6, self.up7)
        u7 = F.relu(self.conv7(torch.cat([c3, u7], dim=1)))
        u8 = _up_k2(u7, self.up8)
        u8 = F.relu(self.conv8(torch.cat([c2, u8], dim=1)))
        u9 = _up_k2(u8, self.up9)
        merged = torch.cat([c1, u9, x], dim=1)
        c9 = F.relu(self.conv9(merged))
        return self.residual(c9)


class Decoder(nn.Module):
    def __init__(self, secret_size=100):
        super().__init__()
        self.secret_size = secret_size
        self.stn = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.stn_fc = nn.Linear(50 * 50 * 128, 128)
        # Zero weights and an identity bias: the first batches are not warped.
        self.stn_w = nn.Parameter(torch.zeros(6, 128))
        self.stn_b = nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.fc1 = nn.Linear(13 * 13 * 128, 512)
        self.fc2 = nn.Linear(512, secret_size)
        self.apply(_he)
        with torch.no_grad():
            self.stn_w.zero_()
            self.stn_b.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def forward(self, image):
        x = image - 0.5
        feat = self.stn(x).flatten(1)
        feat = F.relu(self.stn_fc(feat))
        theta = (feat @ self.stn_w.t() + self.stn_b).view(-1, 2, 3)
        grid = F.affine_grid(theta, x.shape, align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True, padding_mode="zeros")
        x = self.features(x).flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class Critic(nn.Module):
    """WGAN-style critic. A higher score means 'more fake' in this repo's loss."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
        )
        self.apply(_he)

    def forward(self, image):
        score_map = self.net(image - 0.5)
        return score_map.mean()
