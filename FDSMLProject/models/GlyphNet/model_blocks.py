import torch
from torch import nn
from torch.nn import functional as F


class SeparableConv2d(nn.Module):
    """
    Implementazione della convoluzione separabile in PyTorch.

    La convoluzione separabile è una fattorizzazione di una convoluzione pesante in due passaggi più leggeri:
    - Una convoluzione depthwise che applica un filtro separato a ciascun canale di input.
    - Una convoluzione pointwise che applica un filtro 1x1 per combinare i risultati.

    Questo approccio è comunemente utilizzato per ridurre il numero di parametri e il costo computazionale.
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        """
        Inizializza il modulo SeparableConv2d.

        Args:
            in_channels (int): Numero di canali di input.
            out_channels (int): Numero di canali di output.
            kernel_size (int o tuple): Dimensione del kernel della convoluzione.
            bias (bool): Se includere o meno un termine di bias nella convoluzione.
        """
        super(SeparableConv2d, self).__init__()

        # Convoluzione depthwise: applicata separatamente a ciascun canale
        self.depthwise = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                   kernel_size=kernel_size, groups=in_channels, bias=bias, padding=1)

        # Convoluzione pointwise: combina i risultati in un nuovo set di canali
        self.pointwise = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=(1, 1), bias=bias)

    def forward(self, x):
        """
        Passaggio in avanti del modulo SeparableConv2d.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.pointwise(self.depthwise(x))


class FirstBlock(nn.Module):
    """
    Il primo blocco della rete GlyphNet: due blocchi di convoluzione regolari con max-pooling.

    Questo blocco prende un'immagine di input e la converte in una mappa di caratteristiche.
    """

    def __init__(self, in_channels=1,
                 conv_out=64, conv_kernel_size=(3, 3), conv_stride=(1, 1),
                 mp_kernel_size=(3, 3), mp_stride=(2, 2)):
        """
        Inizializza il blocco.

        Args:
            in_channels (int): Numero di canali di input.
            conv_out (int): Numero di canali di output per le convoluzioni.
            conv_kernel_size (tuple): Dimensione del kernel delle convoluzioni.
            conv_stride (tuple): Passo delle convoluzioni.
            mp_kernel_size (tuple): Dimensione del kernel per il max-pooling.
            mp_stride (tuple): Passo per il max-pooling.
        """
        super(FirstBlock, self).__init__()

        # Primo blocco di convoluzione regolare: conv2d + batch_norm + max_pooling
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=conv_out,
                               kernel_size=conv_kernel_size, stride=conv_stride, padding=1)
        self.bn1 = nn.BatchNorm2d(conv_out)
        self.mp1 = nn.MaxPool2d(kernel_size=mp_kernel_size, stride=mp_stride, padding=1)

        # Secondo blocco di convoluzione regolare: conv2d + batch_norm + max_pooling
        self.conv2 = nn.Conv2d(in_channels=conv_out, out_channels=conv_out,
                               kernel_size=conv_kernel_size, stride=conv_stride, padding=1)
        self.bn2 = nn.BatchNorm2d(conv_out)
        self.mp2 = nn.MaxPool2d(kernel_size=mp_kernel_size, stride=mp_stride, padding=1)

    def forward(self, images_tensor):
        """
        Passaggio in avanti del modulo FirstBlock.

        Args:
            images_tensor (torch.Tensor): Tensor dell'immagine di input.

        Returns:
            torch.Tensor: Tensor di output dopo il passaggio attraverso i blocchi convoluzionali.
        """
        first_pass_output = F.relu(self.mp1(self.bn1(self.conv1(images_tensor))))
        second_pass_output = F.relu(self.mp2(self.bn2(self.conv2(first_pass_output))))
        return second_pass_output


class InnerBlock(nn.Module):
    """
    Blocco interno simile a Inception: convoluzioni separabili, normalizzazioni batch, attivazioni e max-pooling.

    Questo blocco viene utilizzato per estrarre ulteriori caratteristiche dall'immagine dopo il blocco iniziale.
    """

    def __init__(self, in_channels, sconv_out=128, sconv_kernel_size=(3, 3), mp_kernel_size=(3, 3), mp_stride=(2, 2)):
        """
        Inizializza il modulo InnerBlock.

        Args:
            in_channels (int): Numero di canali di input.
            sconv_out (int): Numero di canali di output per le convoluzioni separabili.
            sconv_kernel_size (tuple): Dimensione del kernel per le convoluzioni separabili.
            mp_kernel_size (tuple): Dimensione del kernel per il max-pooling.
            mp_stride (tuple): Passo per il max-pooling.
        """
        super(InnerBlock, self).__init__()

        self.sconv1 = SeparableConv2d(in_channels=in_channels, out_channels=sconv_out, kernel_size=sconv_kernel_size)
        self.bn1 = nn.BatchNorm2d(sconv_out)
        self.sconv2 = SeparableConv2d(in_channels=sconv_out, out_channels=sconv_out, kernel_size=sconv_kernel_size)
        self.bn2 = nn.BatchNorm2d(sconv_out)
        self.mp = nn.MaxPool2d(kernel_size=mp_kernel_size, stride=mp_stride, padding=1)

    def forward(self, x):
        """
        Passaggio in avanti del modulo InnerBlock.

        Args:
            x (torch.Tensor): Tensor di input.

        Returns:
            torch.Tensor: Tensor di output dopo il passaggio attraverso i blocchi convoluzionali e max-pooling.
        """
        first_pass_output = F.relu(self.bn1(self.sconv1(x)))
        second_pass_output = F.relu(self.mp(self.bn2(self.sconv2(first_pass_output))))
        return second_pass_output


class FinalBlock(nn.Module):
    """
    Il blocco finale di GlyphNet: convoluzione separabile + global average pooling + dropout + MLP + softmax.

    Questo blocco prepara l'output finale per la classificazione.
    """

    def __init__(self, in_channels=256, out_size=172, sconv_out=512, sconv_kernel_size=(3, 3), dropout_rate=0.15):
        """
        Inizializza il modulo FinalBlock.

        Args:
            in_channels (int): Numero di canali di input.
            out_size (int): Numero di classi di output.
            sconv_out (int): Numero di canali di output per la convoluzione separabile.
            sconv_kernel_size (tuple): Dimensione del kernel per la convoluzione separabile.
            dropout_rate (float): Tasso di dropout per la regolarizzazione.
        """
        super(FinalBlock, self).__init__()
        self.sconv = SeparableConv2d(in_channels=in_channels, out_channels=sconv_out, kernel_size=sconv_kernel_size)
        self.bn = nn.BatchNorm2d(sconv_out)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fully_connected = nn.Linear(in_features=sconv_out, out_features=out_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor):
        """
        Passaggio in avanti del modulo FinalBlock.

        Args:
            input_tensor (torch.Tensor): Tensor di input.

        Returns:
            torch.Tensor: Tensor di output che rappresenta le probabilità logaritmiche delle classi.
        """
        sconv_pass_result = F.relu(self.bn(self.sconv(input_tensor)))

        # Calcolo della media globale su ogni mappa di caratteristiche
        pooled = torch.mean(sconv_pass_result, dim=(-1, -2))  # global average pooling
        return self.softmax(self.fully_connected(self.dropout(pooled)))
