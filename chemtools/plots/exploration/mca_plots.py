# mca_plots.py

import matplotlib.pyplot as plt


def plot_eigenvalues(mca_object):
    """Plots eigenvalues of the MCA."""
    plt.figure(figsize=(8, 6))
    plt.plot(mca_object.V_ordered, marker="o", linestyle="-", color="b")
    plt.title("Scree Plot - Eigenvalues")
    plt.xlabel("Principal Components")
    plt.ylabel("Eigenvalue")
    plt.grid(True)
    plt.show()


def plot_objects(mca_object, axes=[0, 1]):
    """Plots objects on the first two principal components."""
    plt.figure(figsize=(10, 8))
    plt.scatter(
        mca_object.L_ordered[:, axes[0]],
        mca_object.L_ordered[:, axes[1]],
        c=mca_object.objects_colors,
    )
    plt.xlabel(f"PC{axes[0]+1}")
    plt.ylabel(f"PC{axes[1]+1}")
    plt.title("Objects Plot")

    # (Optional) Annotate points with object names
    for i, txt in enumerate(mca_object.objects):
        plt.annotate(
            txt, (mca_object.L_ordered[i, axes[0]], mca_object.L_ordered[i, axes[1]])
        )

    plt.grid(True)
    plt.show()


# Add more plotting functions as needed
# ... (e.g., plot_variables, biplots, etc.)
