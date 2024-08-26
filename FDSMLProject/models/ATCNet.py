import torch
import torch.nn as nn
import torch.nn.functional as F


class ATCNet(nn.Module):
    def __init__(self, n_classes):
        """
        Inizializza l'architettura della rete ATCNet.

        Args:
            n_classes (int): Numero di classi per la classificazione.
        """
        super(ATCNet, self).__init__()

        # Primo strato di convoluzione: input 1 canale, output 64 canali, kernel 3x3
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False)  # Cambiato a 1 canale
        self.bn1 = nn.BatchNorm2d(64)  # Normalizzazione batch dopo la convoluzione
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # Pooling per ridurre le dimensioni spaziali

        # Secondo strato di convoluzione: input 64 canali, output 64 canali, kernel 3x3
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)  # Normalizzazione batch
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # Pooling

        # Primo blocco di convoluzione separabile
        self.sepconv1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False, groups=64),
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False, groups=128),
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # Secondo blocco di convoluzione separabile
        self.sepconv2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False, groups=128),
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False, groups=128),
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # Terzo blocco di convoluzione separabile
        self.sepconv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False, groups=128),
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False, groups=256),
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # Quarto blocco di convoluzione separabile
        self.sepconv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False, groups=256),
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False, groups=256),
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # Blocco di uscita
        self.exit_block = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False, groups=256),
            nn.Conv2d(512, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        # Pooling globale per ridurre l'output a un singolo valore per canale
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.15)
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        """
        Definisce il passaggio in avanti (forward pass) del modello.

        Args:
            x (Tensor): Input tensor, di forma (batch_size, 1, altezza, larghezza).

        Returns:
            Tensor: Logits del modello, di forma (batch_size, n_classes).
        """
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.sepconv1(x)
        x = self.sepconv2(x)
        x = self.sepconv3(x)
        x = self.sepconv4(x)
        x = self.exit_block(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
