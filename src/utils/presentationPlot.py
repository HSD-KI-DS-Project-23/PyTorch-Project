import matplotlib.pyplot as plt


def presentation_plot(image, encoded):
    """Function which plots a MNIST picture, its encoded variant and the predicted label"""

    fig, axs = plt.subplots(1, 2)

    image = image.detach().numpy()
    encoded = encoded.detach().numpy()

    # Plot the MNIST picture
    axs[0].imshow(image.squeeze(), cmap="gray")
    axs[0].axis("off")
    axs[0].set_title("MNIST Picture")

    # Plot the decoded tensor
    axs[1].imshow(encoded.squeeze(), cmap="gray")
    axs[1].axis("off")
    axs[1].set_title("Encoded Tensor")

    # Plot the predicted label
    # axs[2].bar(range(10), label.squeeze())
    # axs[2].set_title("Predicted Label")
    # axs[2].set_xlabel("Digit")
    # axs[2].set_ylabel("Probability")

    plt.tight_layout()
    plt.show()
