import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        
        def double_conv(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.enc1 = double_conv(in_channels, 16)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = double_conv(16, 32)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.bottleneck = double_conv(32, 64)
        
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = double_conv(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = double_conv(32, 16)
        
        self.final_conv = nn.Conv2d(16, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        
        b = self.bottleneck(self.pool2(e2))
        
        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return torch.sigmoid(self.final_conv(d1))

model = UNet()
inputs = torch.randn(4, 1, 64, 64)
masks = torch.randint(0, 2, (4, 1, 64, 64)).float()

criterion = nn.BCELoss()
outputs = model(inputs)
loss = criterion(outputs, masks)

print("UNet Architecture Forward Pass Successful.")
print(f"Input Shape:  {inputs.shape}")
print(f"Output Shape: {outputs.shape}")
print(f"Initial Binary Cross Entropy Loss: {loss.item():.4f}")
