import matplotlib.pyplot as plt
import numpy as np


def view_reconstructed(image, reconstructed):
    """Function for displaying an image (as a PyTorch Tensor) and its
    reconstruction (also a PyTorch Tensor)"""

    if image.shape[1] == 1:
        # MNIST tensor (grayscale image)
        input_image = image[0].squeeze().to("cpu")
        rec_image = reconstructed[0].detach().squeeze().to("cpu").numpy()
        cmap = "gray"
    else:
        # CIFAR10 tensor (RGB image)
        input_image = np.transpose(image[0].to("cpu"), (1, 2, 0))
        rec_image = np.transpose(reconstructed[0].detach().to("cpu").numpy(), (1, 2, 0))
        cmap = None

    # Show input and reconstructed images side by side
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    axes[0].imshow(input_image, cmap=cmap)
    axes[0].axis("off")
    axes[0].set_title("Input Image")
    axes[1].imshow(rec_image, cmap=cmap)
    axes[1].axis("off")
    axes[1].set_title("Reconstructed Image")
    plt.tight_layout()
    plt.show()
    plt.close()
