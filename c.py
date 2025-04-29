# Import necessary libraries
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import colorsys
import random
import math  # For ceiling function in complementary rule
import numpy as np  # For plotting and linspace

# --- Helper Functions ---
# These functions help convert between color formats.


def rgb_to_hex(rgb):
    """Converts RGB tuple (values between 0 and 1) to a hex string."""
    # Ensure values are clipped between 0 and 1 before conversion
    rgb_clipped = tuple(max(0.0, min(1.0, val)) for val in rgb)
    return mc.to_hex(rgb_clipped)


def hex_to_rgb(hex_code):
    """Converts a hex color string (e.g., '#RRGGBB') to an RGB tuple (values between 0 and 1)."""
    try:
        return mc.to_rgb(hex_code)
    except ValueError:
        print(
            f"Warning: Invalid hex code '{hex_code}'. Using black ('#000000') as fallback."
        )
        return (0.0, 0.0, 0.0)


def hls_to_hex(h, l, s):
    """Converts HLS tuple (values between 0 and 1) to a hex string."""
    # Ensure HLS values are within the valid range [0, 1]
    h = h % 1.0  # Hue wraps around
    l = max(0.0, min(1.0, l))
    s = max(0.0, min(1.0, s))
    return rgb_to_hex(colorsys.hls_to_rgb(h, l, s))


def format_rgb_label(rgb_float):
    """Formats RGB float tuple (0-1) into integer string '(R, G, B)' (0-255)."""
    rgb_int = tuple(int(round(c * 255)) for c in rgb_float)
    return f"({rgb_int[0]}, {rgb_int[1]}, {rgb_int[2]})"


# --- The Main Class ---


class HarmonizedPaletteGenerator:
    """
    Generates harmonized color palettes with various styles and options,
    attempting to ensure unique colors even for large requests.

    Attributes:
        n_colors (int): The desired number of colors in the palette.
        style (str): The generation style (e.g., 'normal', 'pastel', 'neon',
                     'light', 'dark', 'contrast', 'husl', 'manual').
        kwargs (dict): Additional arguments specific to certain styles (e.g.,
                       'base_color', 'h', 's', 'l', 'rule', 'angle').
        palette (list | None): Stores the generated list of hex color codes,
                               or None if not generated yet.
    """

    def __init__(self, n_colors, style="normal", **kwargs):
        """
        Initializes the HarmonizedPaletteGenerator.

        Args:
            n_colors (int): The desired number of colors. Must be > 0.
            style (str): The palette style. Defaults to 'normal'.
            **kwargs: Additional keyword arguments for specific styles:
                - base_color (str): Optional hex code for 'light', 'dark', 'manual'.
                                    If not provided, a random base color will be used.
                - h (float): Hue (0-1) for 'husl', 'neon' styles.
                - s (float): Saturation (0-1) for 'husl', 'neon' styles.
                - l (float): Lightness (0-1) for 'husl', 'neon' styles.
                - rule (str): Harmony rule for 'manual' style (e.g.,
                              'monochromatic', 'analogous', 'complementary', 'triadic').
                - angle (float): Angle (degrees) for 'analogous', 'split_complementary'.
                - min_l (float): Minimum lightness (0-1) for 'monochromatic', 'complementary'.
                - max_l (float): Maximum lightness (0-1) for 'monochromatic', 'complementary'.
                - continuous_map (str): Colormap to use if n_colors exceeds qualitative limits
                                        (e.g., 'viridis', 'plasma', 'hsv'). Defaults to 'viridis'.
        """
        if not isinstance(n_colors, int) or n_colors <= 0:
            raise ValueError("n_colors must be a positive integer.")

        self.n_colors = n_colors
        self.style = style.lower()
        self.kwargs = kwargs
        self.palette = None  # Cache for the generated palette

        # Validate arguments early and set random base color if needed
        self._validate_and_prepare_args()

    def _validate_and_prepare_args(self):
        """Checks arguments and sets a random base color if required and not provided."""
        # Styles potentially requiring a base color
        if self.style in ["light", "dark", "manual"]:
            if "base_color" not in self.kwargs or not self.kwargs["base_color"]:
                random_rgb = (random.random(), random.random(), random.random())
                self.kwargs["base_color"] = rgb_to_hex(random_rgb)
                print(
                    f"Info: Style '{self.style}' needs a base color. None provided, using random base: {self.kwargs['base_color']}"
                )
            elif not (
                isinstance(self.kwargs["base_color"], str)
                and self.kwargs["base_color"].startswith("#")
                and len(self.kwargs["base_color"]) == 7
            ):
                raise ValueError(
                    f"Invalid 'base_color' format: {self.kwargs['base_color']}. Must be '#RRGGBB'."
                )

        # HUSL style requirements
        if self.style == "husl":
            if not all(k in self.kwargs for k in ["h", "s", "l"]):
                raise ValueError(
                    "Style 'husl' requires 'h', 's', and 'l' arguments (0-1 range)."
                )
            for k in ["h", "s", "l"]:
                if not isinstance(self.kwargs[k], (int, float)) or not (
                    0 <= self.kwargs[k] <= 1
                ):
                    raise ValueError(
                        f"'{k}' for HUSL must be a number between 0 and 1."
                    )

        # Manual style requirements
        if self.style == "manual":
            if "rule" not in self.kwargs:
                raise ValueError(
                    "Style 'manual' requires a 'rule' argument (e.g., 'monochromatic')."
                )
            valid_rules = [
                "monochromatic",
                "analogous",
                "complementary",
                "triadic",
                "split_complementary",
            ]
            if self.kwargs["rule"] not in valid_rules:
                raise ValueError(
                    f"Invalid 'rule': {self.kwargs['rule']}. Valid rules are: {valid_rules}"
                )

    def generate(self):
        """
        Generates the color palette ensuring unique colors where possible.
        Caches the result in self.palette. Returns the palette list.
        """
        if self.palette is not None:  # Return cached palette if available
            return self.palette

        base_color = self.kwargs.get("base_color")
        continuous_map_name = self.kwargs.get(
            "continuous_map", "viridis"
        )  # Fallback map

        # --- Style-based Generation Logic ---
        generated_palette = []  # Use a temporary list
        try:
            # --- Styles based on predefined palettes ---
            if self.style in ["normal", "pastel", "contrast"]:
                palette_map = {
                    "normal": "tab10",
                    "pastel": "Pastel1",
                    "contrast": "Set1",
                }
                base_palette_name = palette_map[self.style]

                # Adjust contrast palette name based on n_colors
                if self.style == "contrast":
                    if self.n_colors > 9:
                        base_palette_name = "tab20"
                    if self.n_colors > 20:
                        base_palette_name = "tab20b"
                    if self.n_colors > 20:
                        base_palette_name = "tab20c"  # Even more

                try:
                    # Try getting the discrete palette first
                    cmap = plt.get_cmap(base_palette_name)
                    limit = (
                        cmap.N if hasattr(cmap, "N") else self.n_colors
                    )  # N for discrete maps

                    if self.n_colors <= limit:
                        # Use the discrete palette directly if enough colors
                        generated_palette = [
                            mc.to_hex(cmap(i)) for i in range(self.n_colors)
                        ]
                    else:
                        # Fallback to sampling a continuous map if n_colors > limit
                        print(
                            f"Warning: n_colors ({self.n_colors}) exceeds limit ({limit}) for '{base_palette_name}'. "
                            f"Sampling from '{continuous_map_name}' to ensure uniqueness."
                        )
                        cmap_cont = plt.get_cmap(continuous_map_name)
                        generated_palette = [
                            mc.to_hex(cmap_cont(x))
                            for x in np.linspace(0, 1, self.n_colors)
                        ]

                except ValueError:  # Handle case where base_palette_name isn't found
                    print(
                        f"Warning: Could not find palette '{base_palette_name}'. "
                        f"Sampling from '{continuous_map_name}'."
                    )
                    cmap_cont = plt.get_cmap(continuous_map_name)
                    generated_palette = [
                        mc.to_hex(cmap_cont(x))
                        for x in np.linspace(0, 1, self.n_colors)
                    ]

            # --- Styles based on generation functions ---
            elif self.style == "neon":
                # HUSL is good at generating distinct colors
                generated_palette = self._generate_neon()
            elif self.style == "light":
                # Seaborn handles n_colors by interpolation, usually unique
                generated_palette = self._generate_seaborn_light_dark(
                    base_color, dark=False
                )
            elif self.style == "dark":
                # Seaborn handles n_colors by interpolation, usually unique
                generated_palette = self._generate_seaborn_light_dark(
                    base_color, dark=True
                )
            elif self.style == "husl":
                # HUSL is good at generating distinct colors
                h = self.kwargs["h"]
                s = self.kwargs["s"]
                l = self.kwargs["l"]
                generated_palette = self._generate_seaborn_husl(h, s, l)
            elif self.style == "manual":
                # Manual rules now handle uniqueness for large n_colors internally
                rule = self.kwargs["rule"]
                generated_palette = self._generate_manual(base_color, rule)
            else:
                # Fallback for unknown style
                print(
                    f"Warning: Unknown style '{self.style}'. Defaulting to sampling '{continuous_map_name}'."
                )
                self.style = continuous_map_name  # Correct the style name
                cmap_cont = plt.get_cmap(continuous_map_name)
                generated_palette = [
                    mc.to_hex(cmap_cont(x)) for x in np.linspace(0, 1, self.n_colors)
                ]

        except Exception as e:
            print(f"Error generating palette for style '{self.style}': {e}")
            print("Falling back to default black palette.")
            generated_palette = ["#000000"] * self.n_colors  # Fallback palette

        # --- Final Check for Uniqueness (Optional but safe) ---
        if len(set(generated_palette)) < len(generated_palette):
            print(
                "Warning: Duplicate colors detected after generation. Attempting to resolve..."
            )
            # Simple strategy: if duplicates found, fall back to linspace sampling
            # Could implement more sophisticated replacement later if needed
            cmap_cont = plt.get_cmap(continuous_map_name)
            generated_palette = [
                mc.to_hex(cmap_cont(x)) for x in np.linspace(0, 1, self.n_colors)
            ]
            if len(set(generated_palette)) < len(generated_palette):
                print(
                    "Error: Could not resolve duplicates even with fallback sampling."
                )
                # Keep the duplicated list, but warn user

        # --- Finalize and Cache ---
        # Ensure the palette has the requested number of colors (should be guaranteed now)
        if len(generated_palette) != self.n_colors:
            print(
                f"Warning: Final palette size ({len(generated_palette)}) doesn't match requested ({self.n_colors}). This shouldn't happen."
            )
            # Pad or truncate as a last resort
            if len(generated_palette) < self.n_colors:
                last_color = generated_palette[-1] if generated_palette else "#CCCCCC"
                generated_palette.extend(
                    [last_color] * (self.n_colors - len(generated_palette))
                )
            else:
                generated_palette = generated_palette[: self.n_colors]

        self.palette = generated_palette  # Cache the final palette
        return self.palette

    # --- Private Helper Methods for Generation ---

    def _generate_seaborn_light_dark(self, base_color_hex, dark=False):
        """Generates light or dark palettes using Seaborn."""
        hex_to_rgb(base_color_hex)  # Validate hex
        # Seaborn's light/dark palettes interpolate well for uniqueness
        if dark:
            return sns.dark_palette(
                base_color_hex, n_colors=self.n_colors, as_cmap=False
            ).as_hex()
        else:
            return sns.light_palette(
                base_color_hex, n_colors=self.n_colors, as_cmap=False
            ).as_hex()

    def _generate_seaborn_husl(self, h, s, l):
        """Generates HUSL palettes using Seaborn."""
        # HUSL naturally produces distinct colors across the hue range
        return sns.husl_palette(n_colors=self.n_colors, h=h, s=s, l=l).as_hex()

    def _generate_neon(self):
        """Generates 'neon'-like palettes (high saturation/lightness)."""
        h = self.kwargs.get("h", random.random())
        s = self.kwargs.get("s", 0.95)  # Keep high saturation
        l = self.kwargs.get("l", 0.65)  # Keep high lightness
        print(f"Generating 'neon' style using HUSL: h={h:.2f}, s={s:.2f}, l={l:.2f}")
        # Use HUSL generation which handles uniqueness well
        return sns.husl_palette(n_colors=self.n_colors, h=h, s=s, l=l).as_hex()

    def _generate_manual(self, base_hex, rule):
        """Generates palettes using colorsys based on harmony rules, ensuring uniqueness."""
        base_rgb = hex_to_rgb(base_hex)
        base_h, base_l, base_s = colorsys.rgb_to_hls(*base_rgb)
        palette_hls = []

        min_l = self.kwargs.get("min_l", 0.2)  # Adjusted range slightly
        max_l = self.kwargs.get("max_l", 0.9)
        angle = self.kwargs.get("angle", 30.0)

        if rule == "monochromatic":
            # Vary Lightness (L) across the full range
            # np.linspace ensures distinct L values if n_colors > 1
            lightness_values = (
                np.linspace(min_l, max_l, self.n_colors)
                if self.n_colors > 1
                else [base_l]
            )
            for l in lightness_values:
                palette_hls.append((base_h, l, base_s))

        elif rule == "analogous":
            # Vary Hue (H) slightly around the base
            angle_rad_half = (angle / 2.0) / 360.0
            # np.linspace ensures distinct hues if n_colors > 1
            hue_values = (
                np.linspace(
                    base_h - angle_rad_half, base_h + angle_rad_half, self.n_colors
                )
                if self.n_colors > 1
                else [base_h]
            )
            for h in hue_values:
                palette_hls.append((h % 1.0, base_l, base_s))  # Wrap hue

        elif rule == "complementary":
            # Base + Complement + intermediate shades
            comp_h = (base_h + 0.5) % 1.0
            n_base = math.ceil(self.n_colors / 2.0)
            n_comp = self.n_colors // 2

            # Generate unique shades for base color
            base_l_values = (
                np.linspace(min_l, max_l, n_base) if n_base > 1 else [base_l]
            )
            for l in base_l_values:
                palette_hls.append((base_h, l, base_s))

            # Generate unique shades for complementary color
            # Slightly offset lightness range for complement for more distinction?
            comp_min_l = min_l * 1.1
            comp_max_l = max_l * 0.9
            comp_l_values = (
                np.linspace(comp_min_l, comp_max_l, n_comp) if n_comp > 1 else [base_l]
            )
            for l in comp_l_values:
                palette_hls.append((comp_h, l, base_s))

        elif rule in ["split_complementary", "triadic"]:
            # Rules with 3 base hues
            num_base_hues = 3
            if rule == "split_complementary":
                comp_h = (base_h + 0.5) % 1.0
                angle_rad = angle / 360.0
                hues = [base_h, (comp_h - angle_rad) % 1.0, (comp_h + angle_rad) % 1.0]
            else:  # triadic
                hues = [base_h, (base_h + 1 / 3) % 1.0, (base_h + 2 / 3) % 1.0]

            # Distribute n_colors among these hues, varying L/S for uniqueness
            for i in range(self.n_colors):
                h = hues[i % num_base_hues]
                # Calculate cycle number to vary L/S
                cycle = i // num_base_hues
                # Example variation: oscillate lightness slightly around base_l
                # Adjust magnitude and frequency of oscillation as needed
                l_variation = (cycle * 0.08) * (
                    (-1) ** cycle
                )  # Small variation per cycle
                current_l = base_l + l_variation
                # Could also vary saturation slightly:
                # s_variation = (cycle * 0.05) * ((-1)**(cycle+1))
                # current_s = base_s + s_variation
                current_s = base_s  # Keep saturation constant for now

                # Clamp L and S to valid [0, 1] range
                current_l = max(0.05, min(0.95, current_l))  # Avoid pure black/white
                current_s = max(0.1, min(1.0, current_s))  # Avoid pure grey

                palette_hls.append((h, current_l, current_s))

        # --- Convert HLS back to HEX ---
        # Final check for duplicates within the manual generation itself
        hex_palette = [hls_to_hex(h, l, s) for h, l, s in palette_hls]
        if len(set(hex_palette)) < len(hex_palette):
            print(
                f"Warning: Duplicate colors generated within manual rule '{rule}'. "
                "Consider adjusting parameters (min/max L/S, angle) or reducing n_colors."
            )
            # Could attempt a fix here, e.g., jittering duplicates slightly
        return hex_palette

    def _plot_palette_with_labels(self, palette):
        """Internal helper to plot the palette with Hex/RGB labels."""
        n = len(palette)
        fig_width = max(6, n * 1.0)
        fig_height = 2.5
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))

        rgb_palette = [hex_to_rgb(color) for color in palette]
        image_data = np.array(rgb_palette).reshape(1, n, 3)
        ax.imshow(image_data, aspect="auto")

        for i, color_hex in enumerate(palette):
            rgb_float = hex_to_rgb(color_hex)
            rgb_label = format_rgb_label(rgb_float)
            label_text = f"{color_hex}\n{rgb_label}"
            ax.text(
                i,
                1.05,
                label_text,
                ha="center",
                va="bottom",
                color="black",
                fontsize=9,
                transform=ax.transData,
            )

        ax.set_xticks(np.arange(n))
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.tick_params(axis="x", length=0)

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])

        title = f"{self.style.capitalize()} Palette ({self.n_colors} colors)"
        if "base_color" in self.kwargs:
            title += f"\nBase: {self.kwargs['base_color']}"
        fig.text(0.5, 0.97, title, ha="center", va="top", fontsize=12)

        plt.show()

    def get_palette(self, output_format="list"):
        """
        Generates, optionally plots, and optionally returns the color palette list.

        Args:
            output_format (str): Defines the output/action. Options:
                - 'list': Generate and return only the list of hex codes. (Default)
                - 'plot': Generate and display only the palette plot with labels. Returns None.
                - 'list_plot': Generate, display the plot, AND return the list.

        Returns:
            list or None: The list of hex color codes if format is 'list' or
                          'list_plot'. Returns None if format is 'plot' or
                          if palette generation failed.
        """
        # Ensure the palette is generated (or retrieved from cache)
        palette = self.generate()  # Generation logic now ensures uniqueness

        if palette is None:  # Check if generation failed
            print("Error: Palette could not be generated.")
            return None

        output_format = output_format.lower()
        show_plot = output_format in ["plot", "list_plot"]
        return_list = output_format in [
            "list",
            "list_plot",
        ]  # Determine if list should be returned

        # Display the plot with labels if requested
        if show_plot:
            print(f"\n--- Displaying Palette ---")
            try:
                self._plot_palette_with_labels(palette)
            except Exception as e:
                print(f"Error displaying plot: {e}")

        # Return the list ONLY if requested
        if return_list:
            return palette
        else:
            return None  # Return None if format is 'plot'


# --- Example Usage ---
if __name__ == "__main__":

    print("Example 1: Normal Palette (n=15, > tab10 limit)")
    # Expect fallback to sampling 'viridis' (default continuous map)
    gen1 = HarmonizedPaletteGenerator(n_colors=15, style="normal")
    palette1 = gen1.get_palette(output_format="list_plot")
    if palette1:
        print(
            f"Normal (n=15) List length: {len(palette1)}, Unique: {len(set(palette1))}"
        )

    print("\nExample 2: Contrast Palette (n=25, > tab20b limit)")
    # Expect fallback to sampling 'viridis'
    gen2 = HarmonizedPaletteGenerator(n_colors=25, style="contrast")
    palette2 = gen2.get_palette(output_format="plot")  # Returns None

    print("\nExample 3: Manual Triadic (n=10, > 3 base hues)")
    # Expect lightness variation to ensure uniqueness
    gen3 = HarmonizedPaletteGenerator(
        n_colors=10, style="manual", rule="triadic", base_color="#3498db"
    )
    palette3 = gen3.get_palette(output_format="list_plot")
    if palette3:
        print(
            f"Triadic (n=10) List length: {len(palette3)}, Unique: {len(set(palette3))}"
        )

    print("\nExample 4: Manual Split Complementary (n=12, > 3 base hues)")
    gen4 = HarmonizedPaletteGenerator(
        n_colors=12,
        style="manual",
        rule="split_complementary",
        base_color="#e74c3c",
        angle=30,
    )
    palette4 = gen4.get_palette(output_format="list_plot")
    if palette4:
        print(
            f"Split Comp (n=12) List length: {len(palette4)}, Unique: {len(set(palette4))}"
        )

    print("\nExample 5: Monochromatic (n=20)")
    # Uses linspace for lightness, should be unique
    gen5 = HarmonizedPaletteGenerator(
        n_colors=20, style="manual", rule="monochromatic", base_color="#2ecc71"
    )
    palette5 = gen5.get_palette(output_format="plot")  # Returns None

    print("\nExample 6: Using 'hsv' as fallback map for large n")
    gen6 = HarmonizedPaletteGenerator(n_colors=30, style="pastel", continuous_map="hsv")
    palette6 = gen6.get_palette(output_format="list_plot")
    if palette6:
        print(
            f"Pastel (n=30, hsv fallback) List length: {len(palette6)}, Unique: {len(set(palette6))}"
        )
