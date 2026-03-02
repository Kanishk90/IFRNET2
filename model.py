import torch
import torch.nn as nn
import torch.nn.functional as F


def resize(x, scale_factor):
    return F.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=False)


def warp(x, flow):
    B, C, H, W = x.size()
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, H, device=x.device),
        torch.arange(0, W, device=x.device),
        indexing="ij",
    )
    grid = torch.stack((grid_x, grid_y), 2).float()  # H,W,2
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # B,H,W,2
    vgrid = grid + flow.permute(0, 2, 3, 1)

    vgrid_x = 2.0 * vgrid[..., 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[..., 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)

    return F.grid_sample(x, vgrid_scaled, align_corners=False)


def convrelu(in_channels, out_channels, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, k, s, p),
        nn.PReLU(out_channels),
    )


class ResBlock(nn.Module):
    def __init__(self, in_channels, side_channels):
        super().__init__()
        self.side_channels = side_channels
        self.conv1 = convrelu(in_channels, in_channels)
        self.conv2 = convrelu(side_channels, side_channels)
        self.conv3 = convrelu(in_channels, in_channels)
        self.conv4 = convrelu(side_channels, side_channels)
        self.conv5 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.prelu = nn.PReLU(in_channels)

    def forward(self, x):
        out = self.conv1(x)
        out[:, -self.side_channels :, :, :] = self.conv2(
            out[:, -self.side_channels :, :, :].clone()
        )
        out = self.conv3(out)
        out[:, -self.side_channels :, :, :] = self.conv4(
            out[:, -self.side_channels :, :, :].clone()
        )
        return self.prelu(x + self.conv5(out))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.p1 = nn.Sequential(convrelu(3, 32, 3, 2, 1), convrelu(32, 32))
        self.p2 = nn.Sequential(convrelu(32, 48, 3, 2, 1), convrelu(48, 48))
        self.p3 = nn.Sequential(convrelu(48, 72, 3, 2, 1), convrelu(72, 72))
        self.p4 = nn.Sequential(convrelu(72, 96, 3, 2, 1), convrelu(96, 96))

    def forward(self, x):
        f1 = self.p1(x)
        f2 = self.p2(f1)
        f3 = self.p3(f2)
        f4 = self.p4(f3)
        return f1, f2, f3, f4


class Decoder4(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            convrelu(193, 192),
            ResBlock(192, 32),
            nn.ConvTranspose2d(192, 76, 4, 2, 1),
        )

    def forward(self, f0, f1, embt):
        B, _, H, W = f0.shape
        embt = embt.repeat(1, 1, H, W)
        return self.block(torch.cat([f0, f1, embt], 1))


class Decoder3(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            convrelu(220, 216),
            ResBlock(216, 32),
            nn.ConvTranspose2d(216, 52, 4, 2, 1),
        )

    def forward(self, ft, f0, f1, flow0, flow1):
        f0w = warp(f0, flow0)
        f1w = warp(f1, flow1)
        return self.block(torch.cat([ft, f0w, f1w, flow0, flow1], 1))


class Decoder2(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            convrelu(148, 144),
            ResBlock(144, 32),
            nn.ConvTranspose2d(144, 36, 4, 2, 1),
        )

    def forward(self, ft, f0, f1, flow0, flow1):
        f0w = warp(f0, flow0)
        f1w = warp(f1, flow1)
        return self.block(torch.cat([ft, f0w, f1w, flow0, flow1], 1))


class Decoder1(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            convrelu(100, 96),
            ResBlock(96, 32),
            nn.ConvTranspose2d(96, 8, 4, 2, 1),
        )

    def forward(self, ft, f0, f1, flow0, flow1):
        f0w = warp(f0, flow0)
        f1w = warp(f1, flow1)
        return self.block(torch.cat([ft, f0w, f1w, flow0, flow1], 1))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.d4 = Decoder4()
        self.d3 = Decoder3()
        self.d2 = Decoder2()
        self.d1 = Decoder1()

    def inference(self, img0, img1, embt):
        mean_ = torch.cat([img0, img1], 2).mean((1, 2, 3), keepdim=True)
        img0 = img0 - mean_
        img1 = img1 - mean_

        f0 = self.encoder(img0)
        f1 = self.encoder(img1)

        o4 = self.d4(f0[3], f1[3], embt)
        f0u, f1u, ft = o4[:, :2], o4[:, 2:4], o4[:, 4:]

        o3 = self.d3(ft, f0[2], f1[2], f0u, f1u)
        f0u, f1u, ft = o3[:, :2], o3[:, 2:4], o3[:, 4:]

        o2 = self.d2(ft, f0[1], f1[1], f0u, f1u)
        f0u, f1u, ft = o2[:, :2], o2[:, 2:4], o2[:, 4:]

        o1 = self.d1(ft, f0[0], f1[0], f0u, f1u)
        flow0 = o1[:, :2]
        flow1 = o1[:, 2:4]
        mask = torch.sigmoid(o1[:, 4:5])
        res = o1[:, 5:]

        img0w = warp(img0, flow0)
        img1w = warp(img1, flow1)
        out = mask * img0w + (1 - mask) * img1w + mean_ + res
        return torch.clamp(out, 0, 1)
    
    def forward(self, img0, img1, embt, imgt):
        # Predict
        pred = self.inference(img0, img1, embt)

        # Simple L1 reconstruction loss for MRI
        loss_rec = F.l1_loss(pred, imgt)

        # Keep compatibility with your training script
        loss_geo = torch.tensor(0.0, device=img0.device)
        loss_dis = torch.tensor(0.0, device=img0.device)

        return pred, loss_rec, loss_geo, loss_dis