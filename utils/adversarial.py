import torch
from advertorch.attacks import (
    GradientSignAttack,
    LinfBasicIterativeAttack,
    MomentumIterativeAttack,
    PGDAttack
)


class ClassifierWrapper(torch.nn.Module):
    """
    Wrapper for adversarial attacks.

    MAD-VAE forward:
        image -> encoder -> latent -> classifier -> output

    Attacks only need:
        image -> prediction
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output, _, _, _ = self.model(x)
        return output



def add_adv(
        model,
        image,
        label,
        adv,
        epsilon=0.3
):
    """
    Generate adversarial examples.

    Args:
        model:
            MAD-VAE model

        image:
            input images

        label:
            ground truth labels

        adv:
            fgsm / r-fgsm / mi-fgsm / pgd

        epsilon:
            attack strength

    Returns:
        original image,
        adversarial image
    """


    device = image.device


    # Wrap MAD-VAE classifier
    classifier = ClassifierWrapper(model)
    classifier.to(device)
    classifier.eval()


    # -------------------------
    # FGSM
    # -------------------------

    if adv == 'fgsm':

        attack = GradientSignAttack(
            classifier,
            eps=epsilon
        )

        adv_image = attack(
            image,
            label
        )


    # -------------------------
    # Random FGSM
    # -------------------------

    elif adv == 'r-fgsm':

        alpha = epsilon / 2


        noise = torch.empty_like(image).uniform_(
            -alpha,
            alpha
        )

        noisy_image = torch.clamp(
            image + noise,
            0,
            1
        )


        attack = GradientSignAttack(
            classifier,
            eps=epsilon-alpha
        )

        adv_image = attack(
            noisy_image,
            label
        )


    # -------------------------
    # MI-FGSM
    # -------------------------

    elif adv == 'mi-fgsm':

        attack = MomentumIterativeAttack(
            classifier,
            eps=epsilon,
            nb_iter=10,
            eps_iter=epsilon/10
        )

        adv_image = attack(
            image,
            label
        )


    # -------------------------
    # PGD
    # -------------------------

    elif adv == 'pgd':

        attack = PGDAttack(
            classifier,
            eps=epsilon,
            nb_iter=10,
            eps_iter=epsilon/10
        )

        adv_image = attack(
            image,
            label
        )


    # -------------------------
    # No attack
    # -------------------------

    else:

        adv_image = image


    # safety clamp

    adv_image = torch.clamp(
        adv_image,
        0,
        1
    )


    return image, adv_image