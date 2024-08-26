import torch
from torch import nn
from torch.nn import Sequential

from model_blocks import FirstBlock, InnerBlock, FinalBlock


class Glyphnet(nn.Module):
    """
    Implementazione in PyTorch del classificatore di geroglifici GlyphNet.

    Questa classe definisce un modello di rete neurale convoluzionale (CNN) per la classificazione
    dei geroglifici. L'architettura della rete include diversi blocchi convoluzionali, con alcuni
    iperparametri (come il numero di filtri e il tasso di dropout) codificati direttamente per semplicità.

    Attributi:
        first_block (nn.Module): Il blocco convoluzionale iniziale della rete.
        inner_blocks (nn.Sequential): Una sequenza di blocchi convoluzionali intermedi.
        final_block (nn.Module): Il blocco convoluzionale finale che produce l'output.
    """

    def __init__(self, in_channels=1,
                 num_classes=171,
                 first_conv_out=64,
                 last_sconv_out=512,
                 sconv_seq_outs=(128, 128, 256, 256),
                 dropout_rate=0.15):
        """
        Inizializza il modello GlyphNet con gli iperparametri specificati.

        Args:
            in_channels (int): Numero di canali di input (es. 1 per immagini in scala di grigi).
            num_classes (int): Numero di classi per la classificazione finale.
            first_conv_out (int): Numero di canali di output nel primo blocco convoluzionale.
            last_sconv_out (int): Numero di canali di output nel blocco finale.
            sconv_seq_outs (tuple): Canali di output per ciascun blocco convoluzionale sequenziale.
            dropout_rate (float): Tasso di dropout per la regolarizzazione nel blocco finale.
        """
        super(Glyphnet, self).__init__()

        # Definizione del primo blocco della rete
        self.first_block = FirstBlock(in_channels, first_conv_out)

        # Definizione dei blocchi intermedi sequenziali
        in_channels_sizes = [first_conv_out] + list(sconv_seq_outs)
        self.inner_blocks = Sequential(*(InnerBlock(in_channels=i, sconv_out=o)
                                         for i, o in zip(in_channels_sizes, sconv_seq_outs)))

        # Definizione del blocco finale che produce l'output
        self.final_block = FinalBlock(in_channels=in_channels_sizes[-1], out_size=num_classes,
                                      sconv_out=last_sconv_out, dropout_rate=dropout_rate)

    def forward(self, image_input_tensor):
        """
        Definisce il passaggio in avanti del modello.

        Args:
            image_input_tensor (torch.Tensor): Il tensore di input che rappresenta l'immagine.

        Returns:
            torch.Tensor: L'output della rete, corrispondente alle predizioni delle classi.
        """
        x = self.first_block(image_input_tensor)  # Passaggio attraverso il primo blocco
        x = self.inner_blocks(x)  # Passaggio attraverso i blocchi intermedi
        x = self.final_block(x)  # Passaggio attraverso il blocco finale

        return x


if __name__ == "__main__":
    model = Glyphnet()

    print("...la rete proposta ha un numero di parametri molto inferiore, "
          "che è solo 498856 (di cui 494504 sono addestrabili), rispetto a...")

    print("Totale:", sum(p.numel() for p in model.parameters()))
    print("Addestrabili:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    dummy_input = torch.zeros((256, 1, 100, 100))  # batch, immagine a singolo canale
    result = model(dummy_input)

    print(result.shape)
