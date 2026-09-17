import torch
import torch.nn as nn

# -------------------------
# ConvBlock
# -------------------------
class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True)

        )

    def forward(self, x):

        return self.conv(x)
    
# -------------------------
# Encoder
# -------------------------
class Encoder(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):

        feature = self.conv(x)
        pooled = self.pool(feature)

        return feature, pooled
    
# =====================================
# Decoder
# =====================================

class Decoder(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

        self.conv = ConvBlock(
            out_channels * 2,
            out_channels
        )

    def forward(self, x, skip):

        x = self.up(x)

        x = torch.cat(
            [x, skip],
            dim=1
        )

        x = self.conv(x)

        return x
    
# =====================================
# U-Net
# =====================================

class UNet(nn.Module):

    def __init__(self):

        super().__init__()

        # Encoder
        self.encoder1 = Encoder(1, 32)
        self.encoder2 = Encoder(32, 64)
        self.encoder3 = Encoder(64, 128)
        self.encoder4 = Encoder(128, 256)

        # Bottleneck
        self.bottleneck = ConvBlock(256, 512)
        
        # Decoder
        self.decoder4 = Decoder(512, 256)
        self.decoder3 = Decoder(256, 128)
        self.decoder2 = Decoder(128, 64)
        self.decoder1 = Decoder(64, 32)
        
        # 最終出力層
        self.final = nn.Conv2d(
            32,
            1,
            kernel_size=1
        )
        
    def forward(self, x):
        
        # Encoder
        f1, p1 = self.encoder1(x)
        f2, p2 = self.encoder2(p1)
        f3, p3 = self.encoder3(p2)
        f4, p4 = self.encoder4(p3)

        # Bottleneck
        b = self.bottleneck(p4)
        
        # Decoder
        d4 = self.decoder4(b, f4)
        d3 = self.decoder3(d4, f3)
        d2 = self.decoder2(d3, f2)
        d1 = self.decoder1(d2, f1)

        # 出力
        out = self.final(d1)

        return out
        
         
    
if __name__ == "__main__":

    x = torch.randn(1, 1, 256, 256)

    model = UNet()

    y = model(x)

    print("入力:", x.shape)
    print("出力:", y.shape)