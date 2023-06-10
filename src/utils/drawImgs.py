import matplotlib.pyplot as plt


def view_reconstructed(image, reconstructed):
    """Function for displaying an image (as a PyTorch Tensor) and its
    reconstruction also a PyTorch Tensor
    """

    # Show input and reconstructed images side by side
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    axes[0].imshow(image[0].reshape(28, 28).to("cpu"), cmap="gray")
    axes[0].axis("off")
    axes[0].set_title("Input Image")
    axes[1].imshow(
        reconstructed[0].detach().to("cpu").numpy().reshape(28, 28), cmap="gray"
    )
    axes[1].axis("off")
    axes[1].set_title("Reconstructed Image")
    plt.tight_layout()
    # plt.savefig(os.path.join("output", f"epoch_{epoch}.png"))  # Save the figure
    plt.show()
    plt.close()
