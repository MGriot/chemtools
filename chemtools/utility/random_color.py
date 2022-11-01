def random_colorRGB(n):  # random color in RGB
    import random

    color = []
    for _ in range(n):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        rgb = [r, g, b]
        color.append(rgb)
    return color


def random_colorHEX(n):  # random color in RGB
    import random

    color = []
    for _ in range(n):
        hexadecimal = [
            "#" + "".join([random.choice("ABCDEF0123456789") for _ in range(6)])
        ]
        color.append(hexadecimal)
    return color
