import random


def random_colorRGB(n):
    """Generate random RGB color

    Args:
        n (number): Number of color that you want.

    Returns:
        list: List of colors
    """
    colors = []
    for _ in range(n):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        rgb = (r, g, b)  # Usa una tupla invece di una lista
        colors.append(rgb)
    return colors


def random_colorHEX(n):
    """Generates a list of n random HEX color codes."""
    color_list = [
        "#" + "".join([random.choice("0123456789ABCDEF") for j in range(6)])
        for i in range(n)
    ]
    return color_list
