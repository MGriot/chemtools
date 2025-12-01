import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from typing import Dict, List, Tuple

from ..base import BasePlotter

class SunburstPlot(BasePlotter):
    """
    A plotter for creating layered sunburst charts.
    """

    def plot(
        self,
        df: pd.DataFrame,
        cols: List[str],
        count_col: str = 'Material',
        status_col: str = 'Status',
        status_ok_val: str = 'Ok',
        start_angle: int = 90,
        top_n_limits: Tuple[int, int] = (6, 10), # (Top N for Layer 1, Top N for Layer 2)
        label_color: str = 'black',
        **kwargs
    ):
        """
        Generates a 3-layer Sunburst chart (Layer 1 -> Layer 2 -> Status).

        Parameters:
        - df: The Pandas DataFrame (already filtered for Shims/Items of interest).
        - cols: List of 2 column names representing the hierarchy [Layer1, Layer2] 
                (e.g., ['Market', 'Meta-Supplier']).
        - count_col: The column to count unique values of (e.g., 'Material').
        - status_col: Column determining transparency (Solid vs Transparent).
        - status_ok_val: The value in status_col that represents "Robust/Solid".
        - start_angle: Rotation of the chart.
        - top_n_limits: Tuple (Top N for Layer 1, Top N for Layer 2).
        - label_color: Color of the text for outside labels.
        - **kwargs: Additional keyword arguments passed to the BasePlotter.
                    Can include 'title', 'subplot_title', 'figsize', etc.
        """
        params = self._process_common_params(**kwargs)
        
        # Unpack column names for clarity
        col_l1, col_l2 = cols
        
        # --- STEP 1: DATA PREPARATION & HANDLING "OTHERS" ---
        df_work = df.copy()
        
        # Helper to group small categories into "Other"
        def group_others(dframe, column, limit):
            counts = dframe.groupby(column)[count_col].nunique().sort_values(ascending=False)
            top_list = counts.head(limit).index.tolist()
            return dframe[column].apply(lambda x: x if x in top_list else 'Other')

        # Apply grouping
        df_work[f'Display_{col_l1}'] = group_others(df_work, col_l1, top_n_limits[0])
        df_work[f'Display_{col_l2}'] = group_others(df_work, col_l2, top_n_limits[1])
        
        # Display Names
        disp_l1 = f'Display_{col_l1}'
        disp_l2 = f'Display_{col_l2}'

        # --- STEP 2: AGGREGATION ---
        # Group by L1, L2, Status
        grouped = df_work.groupby([disp_l1, disp_l2, status_col])[count_col].nunique().reset_index()
        grouped.rename(columns={count_col: 'Count'}, inplace=True)
        
        # Sort for visual alignment (Status False usually comes after True for alignment, or specific order)
        # We sort L1 (asc), L2 (asc), Status (desc -> Ok first)
        grouped.sort_values(by=[disp_l1, disp_l2, status_col], ascending=[True, True, False], inplace=True)

        # Prepare Layers
        # 1. Inner (Layer 1)
        l1_data = grouped.groupby(disp_l1)['Count'].sum().reset_index()
        l1_data.sort_values(by=disp_l1, inplace=True)
        
        # 2. Middle (Layer 2)
        l2_data = grouped.groupby([disp_l1, disp_l2])['Count'].sum().reset_index()
        l2_data.sort_values(by=[disp_l1, disp_l2], inplace=True)
        
        # 3. Outer (Status)
        outer_data = grouped.sort_values(by=[disp_l1, disp_l2, status_col], ascending=[True, True, False])
        
        total_count = outer_data['Count'].sum()

        # --- STEP 3: COLOR MAPPING ---
        # Use BasePlotter's category_color_scale, with a fallback if BasePlotter's theme setup failed
        cmap_chemtools = self.colors.get('category_color_scale')
        if not isinstance(cmap_chemtools, list) or not cmap_chemtools:
            # Fallback to a default matplotlib colormap
            cmap_chemtools = plt.cm.get_cmap('tab10').colors.tolist()
            if not cmap_chemtools: # Defensive check
                raise ValueError("Could not get a valid color map for plotting.")

        unique_l1_vals = l1_data[disp_l1].unique()
        
        # Ensure cmap_chemtools is not empty to avoid ZeroDivisionError in modulo operation
        if not cmap_chemtools:
            raise ValueError("Color map is empty, cannot assign colors to categories.")

        l1_color_map = {val: cmap_chemtools[i % len(cmap_chemtools)] for i, val in enumerate(unique_l1_vals)}

        # Generate Color Lists
        inner_colors = [l1_color_map[val] for val in l1_data[disp_l1]]
        middle_colors = [l1_color_map[row[disp_l1]] for _, row in l2_data.iterrows()]
        
        outer_colors = []
        alpha_solid = 1.0
        alpha_trans = 0.20
        
        for idx, row in outer_data.iterrows():
            l1_val = row[disp_l1]
            
            # Initialize with default fallback values
            r_final, g_final, b_final = (0.5, 0.5, 0.5) # Gray
            alpha_from_color_map = 1.0 # Opaque

            color_from_map = l1_color_map.get(l1_val)
            if color_from_map is not None:
                try:
                    rgba_tuple = mcolors.to_rgba(color_from_map)
                    if rgba_tuple is not None and len(rgba_tuple) == 4: # Ensure it's a valid 4-tuple
                        r_final, g_final, b_final, alpha_from_color_map = rgba_tuple
                    else:
                        print(f"WARNING: mcolors.to_rgba('{color_from_map}') returned invalid RGBA. Using gray fallback.")
                except ValueError as e:
                    print(f"WARNING: mcolors.to_rgba failed for color '{color_from_map}' (l1_val='{l1_val}'): {e}. Using gray fallback.")
            else:
                print(f"WARNING: l1_val '{l1_val}' not found in l1_color_map. Using gray fallback.")


            if row[status_col] == status_ok_val:
                outer_colors.append((r_final, g_final, b_final, alpha_solid))
            else:
                outer_colors.append((r_final, g_final, b_final, alpha_trans))

        # --- STEP 4: PLOTTING ---
        fig, ax = self._create_figure(figsize=params["figsize"])
        size = 0.3
        
        def make_autopct(pct):
            return f'{pct:.1f}%' if pct > 4 else ''

        # RING 1: Inner
        wedges1, texts1, autotexts1 = ax.pie(
            l1_data['Count'], radius=1 - 2*size, colors=inner_colors,
            wedgeprops=dict(width=size, edgecolor=self.colors['bg_color']), autopct=make_autopct,
            pctdistance=0.75, labels=None, startangle=start_angle
        )
        for t in autotexts1:
            t.set_color(self.colors['bg_color']) # Text on inner ring usually contrasting with segment
            t.set_weight('bold')
            t.set_fontsize(9)

        # RING 2: Middle
        wedges2, texts2, autotexts2 = ax.pie(
            l2_data['Count'], radius=1 - size, colors=middle_colors,
            wedgeprops=dict(width=size, edgecolor=self.colors['bg_color']), autopct='',
            labels=None, startangle=start_angle
        )

        # RING 3: Outer
        wedges3, texts3 = ax.pie(
            outer_data['Count'], radius=1, colors=outer_colors,
            wedgeprops=dict(width=size, edgecolor=self.colors['bg_color']), labels=None,
            startangle=start_angle, autopct=None # Set autopct=None to return only 2 values
        )

        # --- STEP 5: ANNOTATIONS ---
        
        # Middle Ring Annotations
        threshold_l2 = total_count * 0.02
        
        for i, p in enumerate(wedges2):
            ang = (p.theta2 - p.theta1) / 2.0 + p.theta1
            val = l2_data.iloc[i]['Count']
            name = l2_data.iloc[i][disp_l2]
            pct_val = (val / total_count) * 100
            
            # Shorten text
            name_short = name[:10] + '...' if len(name) > 10 else name
            
            # Geometry
            r_inside = 1 - 1.5 * size
            x_in = np.cos(np.deg2rad(ang)) * r_inside
            y_in = np.sin(np.deg2rad(ang)) * r_inside
            rot = ang - 180 if x_in < 0 else ang

            if val > threshold_l2:
                # Inside Label
                ax.text(x_in, y_in, f"{name_short}\n({pct_val:.1f}%)",
                        ha='center', va='center', rotation=rot,
                        fontsize=7.5, fontweight='bold', color=self.colors['bg_color']) # Text color for middle ring
            else:
                # Outside Label (Leader Line)
                if pct_val >= 0.8:
                    r_out = 1.3 - size + 0.08
                    x_out = np.cos(np.deg2rad(ang)) * r_out
                    y_out = np.sin(np.deg2rad(ang)) * r_out
                    ha = 'left' if x_out >= 0 else 'right'
                    
                    ax.annotate(
                        f"{name_short} ({pct_val:.1f}%)",
                        xy=(np.cos(np.deg2rad(ang)) * (1 - size/2), np.sin(np.deg2rad(ang)) * (1 - size/2)),
                        xytext=(x_out, y_out), ha=ha, va='center',
                        fontsize=7, fontweight='bold', color=self.colors['text_color'], # label_color can be passed via kwargs
                        arrowprops=dict(arrowstyle='-', color=self.colors['text_color'], lw=0.8, clip_on=False),
                        bbox=dict(facecolor=self.colors['bg_color'], edgecolor='none', pad=1.5)
                    )

        # Outer Ring Annotations (Status)
        threshold_outer = total_count * 0.015
        for i, p in enumerate(wedges3):
            row = outer_data.iloc[i]
            val = row['Count']
            if row[status_col] == status_ok_val and val > threshold_outer:
                ang = (p.theta2 - p.theta1)/2. + p.theta1
                r_text = 1 - size/2
                x = np.cos(np.deg2rad(ang)) * r_text
                y = np.sin(np.deg2rad(ang)) * r_text
                rot = ang - 180 if x < 0 else ang
                pct = (val / total_count) * 100
                ax.text(x, y, f"{pct:.1f}%", ha='center', va='center', rotation=rot,
                        fontsize=6.5, fontweight='bold', color=self.colors['bg_color']) # Text color for outer ring

        # --- STEP 6: LEGEND ---
        # legend_elements = [Patch(facecolor=l1_color_map[m], label=m) for m in unique_l1_vals]
        # legend_elements.append(Patch(facecolor='none', edgecolor='none', label='-----------------')) # Separator
        legend_elements = []
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=f'{status_ok_val} (Solid)',
                                      markerfacecolor=(0.3, 0.3, 0.3, alpha_solid), markersize=10))
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Not (Transparent)',
                                      markerfacecolor=(0.3, 0.3, 0.3, alpha_trans), markersize=10))

        # Pass custom legend handles and labels to _apply_common_layout
        params['legend_handles'] = legend_elements
        params['legend_labels'] = [e.get_label() for e in legend_elements]
        # Set legend title if not already set via legend_opts
        if 'legend_opts' not in params or ('title' not in params['legend_opts'] and 'title' not in self.legend_opts):
            params['legend_opts'] = params.get('legend_opts', {})
            params['legend_opts']['title'] = col_l1
        params['showlegend'] = True # Ensure legend is shown

        # --- STEP 7: Final Layout and Save ---
        self._set_labels(
            ax,
            subplot_title=params.get("subplot_title", f"Breakdown: {col_l1} > {col_l2} > {status_col}"),
            # The main title will be handled by params.get("title") if provided.
        )
        self._apply_common_layout(fig, params)
        
        # The user will call save() or show() on the returned figure object if needed.
        return fig