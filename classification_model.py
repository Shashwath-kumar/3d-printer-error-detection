import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, in_channels):
        super(Attention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        q = self.conv1(x)
        k = self.conv2(x)
        v = self.conv3(x)

        attention_map = F.softmax(torch.matmul(q, k.transpose(2, 3)), dim=-1)
        out = torch.matmul(attention_map, v)

        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, in_channels, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList([Attention(in_channels) for _ in range(num_heads)])

    def forward(self, x):
        outputs = [head(x) for head in self.heads]
        return sum(outputs)

class ImageClassificationModel(nn.Module):
    def __init__(self, num_heads=4):
        super(ImageClassificationModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.multi_head_attention = MultiHeadAttention(256, num_heads)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = self.multi_head_attention(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = torch.sigmoid(x)

        return x.squeeze()

if __name__ == '__main__':
    # Create the model
    model = ImageClassificationModel(num_heads=4)
