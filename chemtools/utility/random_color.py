import random

def random_colorRGB(n):
    """Generate random RGB color

    Args:
        n (number): Number of color that you want.

    Returns:
        array: Array of colors
    """
    colors = []
    for _ in range(n):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        rgb = [r, g, b]
        colors.append(rgb)
    return colors


def random_colorHEX(n):
    """Generate random HEX color

    Args:
        n (number): Number of color that you want.

    Returns:
        array: Array of colors
    """
    colors = []
    for _ in range(n):
        hexadecimal = [
            "#" + "".join([random.choice("ABCDEF0123456789") for _ in range(6)])
        ]
        colors.append(hexadecimal)
    return colors
