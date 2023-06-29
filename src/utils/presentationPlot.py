# Author: Leon Kleinschmidt

import matplotlib.pyplot as plt
import numpy as np


def presentation_plot(plot_title, image, encoded, decoded, label):
    """Function which plots a MNIST picture, its encoded variant, its decoded variant and the predicted label"""

    fig, axs = plt.subplots(1, 4)

    # Set the title for the entire figure
    fig.suptitle(plot_title)

    image = image.detach().numpy()
    encoded = encoded.detach().numpy()
    decoded = decoded.detach().numpy()
    label = label.detach().numpy()

    # Plot the MNIST picture
    axs[0].imshow(image.squeeze(), cmap="gray")
    axs[0].axis("off")
    axs[0].set_title("MNIST Picture")

    # Plot the encoded tensor
    axs[1].imshow(encoded.squeeze(), cmap="gray")
    axs[1].axis("off")
    axs[1].set_title("Encoded Tensor")

    # Plot the decoded tensor
    axs[2].imshow(decoded.squeeze(), cmap="gray")
    axs[2].axis("off")
    axs[2].set_title("Decoded Picture")

    # Plot the predicted label
    axs[3].bar(range(10), label.squeeze())
    axs[3].set_xticks(np.arange(10))
    axs[3].set_title("Predicted Label")
    axs[3].set_xlabel("Digit")
    axs[3].set_ylabel("Probability")

    plt.tight_layout()
    # Save the figure as svg
    # plt.savefig("figure.svg", format="svg")
    plt.show()
