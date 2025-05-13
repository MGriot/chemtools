# -*- coding: utf-8 -*-
"""
This module provides base model functionality for statistical analysis tools.
It contains the BaseModel abstract base class which implements common model operations
and defines the interface for specific model implementations.
"""

import pickle
import datetime
import math
import warnings
import inspect
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional, Union


# --- Constants for Formatting ---
DEFAULT_SUMMARY_WIDTH: int = 90
DEFAULT_SEPARATOR_CHAR: str = "-"
FINAL_SEPARATOR_CHAR: str = "="
DEFAULT_COLUMN_SEPARATOR: str = "  "  # Separator between columns in multicolumn
KEY_VALUE_SEPARATOR: str = (
    ": "  # Separator between key text and value text in multicolumn
)
DEFAULT_NOTES_INDENT: str = "    "
MIN_KEY_VALUE_DISPLAY_WIDTH: int = (
    10  # Minimum width for key text part itself in multicolumn
)


class BaseModel(ABC):
    """
    Abstract base class for statistical models, providing common
    functionalities like saving/loading and a basic summary.

    Attributes:
        model_name (str): Name of the model.
        method (Optional[str]): Name of the statistical method used.
        notes (List[str]): A list of strings containing notes about the model run.
    """

    def __init__(self):
        """Initializes the BaseModel."""
        self.model_name: str = "Base Model"
        self.method: Optional[str] = None
        self.notes: List[str] = []

    def save(self, filename: str):
        """Save model instance to a file using pickle serialization.

        Args:
            filename: Path to save the model to.
        """
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename: str) -> "BaseModel":
        """Load a model instance from a file.

        Args:
            filename: Path to load the model from.

        Returns:
            The loaded model instance.
        """
        with open(filename, "rb") as file:
            loaded_model: "BaseModel" = pickle.load(file)
            return loaded_model

    @abstractmethod
    def _get_summary_data(self) -> Dict[str, Any]:
        """
        Abstract method for subclasses to provide summary data.
        """
        return {}

    def _create_general_summary(
        self, n_variables: int, n_objects: int, **kwargs
    ) -> Dict[str, Dict[str, str]]:
        """Creates the 'general' part of the summary data dictionary."""
        summary_data = {
            "general": {
                "No. Variables": str(n_variables),
                "No. Objects": str(n_objects),
            },
        }
        summary_data["general"].update({k: str(v) for k, v in kwargs.items()})
        return summary_data

    # --- Summary Formatting Helpers ---

    def _format_separator(self, width: int, char: str = DEFAULT_SEPARATOR_CHAR) -> str:
        """Creates a separator line string."""
        return char * width

    def _format_title(self, width: int) -> str:
        """Formats the main title section of the summary."""
        title = f"{self.method or self.model_name or 'Model'} Summary"
        separator = self._format_separator(width, DEFAULT_SEPARATOR_CHAR)
        final_separator = self._format_separator(width, FINAL_SEPARATOR_CHAR)
        return f"{final_separator}\n{title.center(width)}\n{separator}\n"

    def _get_datetime_info(self) -> Dict[str, str]:
        """Gets current date and time formatted for the summary."""
        now = datetime.datetime.now()
        return {
            "Date:": now.strftime("%a, %d %b %Y"),
            "Time:": now.strftime("%H:%M:%S"),
        }

    def _calculate_multicolumn_widths(
        self, data: Dict[str, str], total_width: int, num_cols: int
    ) -> Tuple[int, int, int]:
        """Calculates widths for multi-column dictionary formatting.

        Args:
            data: The dictionary data to be formatted.
            total_width: The total available width for the output.
            num_cols: The number of columns to format into.

        Returns:
            Tuple[int, int, int]: (col_width_per_item, left_block_width, key_text_display_width)
                                   col_width_per_item: general width for columns if divided equally.
                                   left_block_width: width for the entire first column block.
                                   key_text_display_width: width allocated for the key text itself (before KEY_VALUE_SEPARATOR).
        """
        if not data or num_cols <= 0:
            return 0, 0, 0

        col_width_per_item = total_width // num_cols if num_cols > 0 else total_width

        if num_cols == 1:
            left_block_width = total_width
        elif num_cols == 2:
            # For a 2-column layout, a common split. Adjust if needed.
            # This value influences how much space is left for the key text vs value.
            left_block_width = int(
                total_width * 0.48
            )  # Adjusted for potentially longer keys/values in col1
            if (
                left_block_width
                < MIN_KEY_VALUE_DISPLAY_WIDTH + len(KEY_VALUE_SEPARATOR) + 5
            ):  # Min key + sep + min val
                left_block_width = (
                    MIN_KEY_VALUE_DISPLAY_WIDTH + len(KEY_VALUE_SEPARATOR) + 5
                )
            if left_block_width >= total_width - (
                MIN_KEY_VALUE_DISPLAY_WIDTH
                + len(KEY_VALUE_SEPARATOR)
                + 5
                + len(DEFAULT_COLUMN_SEPARATOR)
            ):
                # Ensure right column also has some minimal space
                left_block_width = (
                    total_width
                    - (
                        MIN_KEY_VALUE_DISPLAY_WIDTH
                        + len(KEY_VALUE_SEPARATOR)
                        + 5
                        + len(DEFAULT_COLUMN_SEPARATOR)
                    )
                    - 1
                )

        else:  # num_cols > 2
            left_block_width = col_width_per_item

        try:
            max_key_text_len = (
                max(len(str(k).rstrip(":")) for k in data.keys()) if data else 0
            )
        except ValueError:
            max_key_text_len = 0

        key_text_display_width = max(MIN_KEY_VALUE_DISPLAY_WIDTH, max_key_text_len)

        # Max width for "KeyText: " part in the left column
        max_key_prefix_in_left = left_block_width - 3  # Leave at least 3 for value
        if key_text_display_width + len(KEY_VALUE_SEPARATOR) > max_key_prefix_in_left:
            key_text_display_width = max(
                MIN_KEY_VALUE_DISPLAY_WIDTH,
                max_key_prefix_in_left - len(KEY_VALUE_SEPARATOR),
            )

        if num_cols > 1:
            right_col_block_width = (
                (total_width - left_block_width)
                if num_cols == 2
                else col_width_per_item
            )
            # Max width for "KeyText: " part in a right column cell
            # (cell_width - default_col_sep - min_val_space)
            max_key_prefix_in_right_cell = (
                right_col_block_width - len(DEFAULT_COLUMN_SEPARATOR) - 3
            )
            if (
                key_text_display_width + len(KEY_VALUE_SEPARATOR)
                > max_key_prefix_in_right_cell
            ):
                key_text_display_width = max(
                    MIN_KEY_VALUE_DISPLAY_WIDTH,
                    max_key_prefix_in_right_cell - len(KEY_VALUE_SEPARATOR),
                )

        key_text_display_width = max(
            key_text_display_width, MIN_KEY_VALUE_DISPLAY_WIDTH
        )

        return col_width_per_item, left_block_width, key_text_display_width

    def _format_multicolumn_dict(
        self, data: Dict[str, str], total_width: int, num_cols: int = 2
    ) -> str:
        """Formats a dictionary into a multi-column string.
        Keys are left-aligned. Values in ALL columns are right-aligned within their available space
        after the key prefix. The colon (KEY_VALUE_SEPARATOR) is attached directly to the key text.
        """
        if not data or num_cols <= 0:
            return ""

        col_width_per_item, left_block_width, key_text_disp_width = (
            self._calculate_multicolumn_widths(data, total_width, num_cols)
        )

        right_overall_block_width = total_width - left_block_width

        items = list(data.items())
        num_rows = math.ceil(len(items) / num_cols)
        lines: List[str] = []

        formatted_key_prefix_width = key_text_disp_width + len(KEY_VALUE_SEPARATOR)

        for row_idx in range(num_rows):
            row_parts: List[str] = []
            for col_idx in range(num_cols):
                item_idx = row_idx + col_idx * num_rows
                if item_idx < len(items):
                    key, value = items[item_idx]
                    key_text_original = str(key).rstrip(":")
                    value_str = str(value)

                    key_plus_separator = f"{key_text_original}{KEY_VALUE_SEPARATOR}"
                    key_segment = key_plus_separator.ljust(
                        formatted_key_prefix_width
                    )  # This is "KeyText:      "

                    current_item_cell_total_width: int
                    leading_separator = ""

                    if col_idx == 0:  # First column
                        current_item_cell_total_width = left_block_width
                        # No leading separator for the very first column's content block
                    else:  # Second and subsequent columns
                        current_item_cell_total_width = (
                            right_overall_block_width
                            if num_cols == 2 and col_idx == 1
                            else col_width_per_item
                        )
                        leading_separator = DEFAULT_COLUMN_SEPARATOR

                    # Calculate space for value AFTER key_segment and any leading_separator
                    space_for_value = (
                        current_item_cell_total_width
                        - len(key_segment)
                        - len(leading_separator)
                    )

                    display_value_str = value_str
                    if space_for_value < 1:
                        # Not enough space for value, it might get truncated or just appended if key_segment itself overflows.
                        # If key_segment + leading_separator already fills/overflows cell_width, value is effectively lost or makes line too long.
                        # For simplicity, if no space, value is empty or minimal.
                        display_value_str = (
                            ""
                            if space_for_value < 0
                            else value_str[: max(0, space_for_value)]
                        )
                    elif len(value_str) > space_for_value:
                        if space_for_value >= 3:  # Can we fit "..."?
                            display_value_str = value_str[: space_for_value - 3] + "..."
                        else:  # Can't fit "...", just truncate
                            display_value_str = value_str[:space_for_value]

                    aligned_value = display_value_str.rjust(
                        max(0, space_for_value)
                    )  # Use max(0,..) in case space_for_value became negative

                    part_content = f"{leading_separator}{key_segment}{aligned_value}"

                    # The part_content should ideally fit current_item_cell_total_width.
                    # Ljust to ensure it fills the block if calculations were slightly off or value was empty.
                    row_parts.append(part_content.ljust(current_item_cell_total_width))

                else:  # Empty cell padding
                    empty_part_width = (
                        left_block_width
                        if col_idx == 0
                        else (
                            right_overall_block_width
                            if num_cols == 2 and col_idx == 1
                            else col_width_per_item
                        )
                    )
                    prefix = DEFAULT_COLUMN_SEPARATOR if col_idx > 0 else ""
                    row_parts.append(prefix + (" " * (empty_part_width - len(prefix))))

            lines.append("".join(row_parts).rstrip())

        return "\n".join(lines) + "\n"

    def _calculate_table_col_widths(
        self, headers: List[Any], total_width: int
    ) -> List[int]:
        """Calculates widths for table columns based on headers and total width."""
        num_cols = len(headers)
        if num_cols == 0:
            return []

        separator_space = len(DEFAULT_COLUMN_SEPARATOR) * (num_cols - 1)
        available_width = total_width - separator_space

        if available_width <= 0:
            base_width = max(1, total_width // num_cols) if num_cols > 0 else 0
            widths = [base_width] * num_cols
            rem_after_base = total_width - (sum(widths) + separator_space)
            if rem_after_base < 0:
                rem_after_base = 0

            for i in range(min(rem_after_base, num_cols)):
                widths[i] += 1
            return widths

        base_col_width = available_width // num_cols
        remainder = available_width % num_cols
        col_widths = [base_col_width] * num_cols
        for i in range(remainder):
            col_widths[i] += 1

        return col_widths

    def _format_table(
        self, table_info: Union[List[List[Any]], Dict[str, Any]], total_width: int
    ) -> str:
        """Formats a table (list of lists or dict with data/align) into a fixed-width string."""
        alignments: Optional[List[str]] = None
        table_data: Optional[List[List[Any]]] = None

        if isinstance(table_info, dict):
            table_data = table_info.get("data")
            alignments = table_info.get("align")
        elif isinstance(table_info, list):
            table_data = table_info
        else:
            return "[Error formatting table: Invalid table_info type]\n"

        if (
            not table_data
            or not isinstance(table_data, list)
            or not table_data
            or not isinstance(table_data[0], (list, tuple))
            or not table_data[0]
        ):
            return ""

        headers = table_data[0]
        num_cols = len(headers)
        if num_cols == 0:
            return ""

        col_widths = self._calculate_table_col_widths(headers, total_width)

        if not col_widths or len(col_widths) != num_cols:
            return "[Error formatting table: Width calculation failed or mismatch]\n"

        final_alignments: List[str] = []
        if isinstance(alignments, list) and len(alignments) == num_cols:
            valid_aligns = {"<", ">", "^"}
            if all(a in valid_aligns for a in alignments):
                final_alignments = alignments
            else:
                warnings.warn(
                    "Invalid alignment specifier in table_info['align']. Using default.",
                    UserWarning,
                    stacklevel=2,
                )
                alignments = None

        if not final_alignments:
            final_alignments = ["<"] + [">"] * (num_cols - 1) if num_cols > 0 else []

        lines: List[str] = []
        for row_idx, row_data in enumerate(table_data):
            if not isinstance(row_data, (list, tuple)):
                warnings.warn(
                    f"Row {row_idx} in table is not a list/tuple, skipping.",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            padded_row = list(row_data[:num_cols]) + [""] * (num_cols - len(row_data))
            str_row = [str(cell) for cell in padded_row]

            formatted_cells = []
            for j, cell_str in enumerate(str_row):
                if j >= len(final_alignments) or j >= len(col_widths):
                    warnings.warn(
                        f"Column index {j} out of bounds for alignments/widths in table. Skipping cell.",
                        UserWarning,
                        stacklevel=2,
                    )
                    formatted_cells.append("")
                    continue

                alignment = final_alignments[j]
                max_cell_width = col_widths[j]

                if max_cell_width <= 0:
                    formatted_cells.append("")
                    continue

                display_cell = cell_str
                if len(cell_str) > max_cell_width:
                    if max_cell_width >= 3:
                        display_cell = cell_str[: max_cell_width - 3] + "..."
                    else:
                        display_cell = cell_str[:max_cell_width]

                formatted_cells.append(f"{display_cell:{alignment}{max_cell_width}}")

            lines.append(DEFAULT_COLUMN_SEPARATOR.join(formatted_cells).rstrip())

        return "\n".join(lines) + "\n"

    def _wrap_text(self, text: str, width: int) -> List[str]:
        """Wraps text to a given width, handling line breaks and long words."""
        if width <= 0:
            return text.splitlines() if text else [""]

        lines: List[str] = []
        paragraphs = text.splitlines()
        if not paragraphs and text:
            paragraphs = [text]
        elif not text:
            return [""]

        for para_idx, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                lines.append("")
                continue

            words = paragraph.split()

            current_line = ""
            for word_idx, word in enumerate(words):
                potential_line = (
                    f"{current_line} {word}".strip() if current_line else word
                )

                if len(potential_line) <= width:
                    current_line = potential_line
                else:
                    if current_line:
                        lines.append(current_line)

                    if len(word) > width:
                        start = 0
                        while start < len(word):
                            lines.append(word[start : start + width])
                            start += width
                        current_line = ""
                    else:
                        current_line = word

            if current_line:
                lines.append(current_line)

        return lines if lines else [""]

    def _format_notes(self, total_width: int) -> str:
        """Formats the notes section with numbering and text wrapping."""
        if not self.notes:
            return ""

        output_lines: List[str] = ["Notes:"]
        indent_len = len(DEFAULT_NOTES_INDENT)
        note_text_width = total_width - indent_len
        if note_text_width <= 0:
            note_text_width = max(1, total_width - 1)

        for i, note_item in enumerate(self.notes, 1):
            note_str = str(note_item)
            wrapped_note_lines = self._wrap_text(note_str, note_text_width)

            if not wrapped_note_lines or (
                len(wrapped_note_lines) == 1 and not wrapped_note_lines[0].strip()
            ):
                prefix = f"[{i}] ".ljust(indent_len)
                output_lines.append(f"{prefix}")
                continue

            for j, line_content in enumerate(wrapped_note_lines):
                prefix = f"[{i}] ".ljust(indent_len) if j == 0 else DEFAULT_NOTES_INDENT
                output_lines.append(f"{prefix}{line_content}")

        return (
            "\n".join(output_lines) + "\n"
            if output_lines and output_lines != ["Notes:"]
            else ""
        )

    @property
    def summary(self) -> str:
        """Generates a formatted summary string of the model results."""
        try:
            summary_data = self._get_summary_data()
            if not isinstance(summary_data, dict):
                detail = f"Expected dict, got {type(summary_data).__name__}."
                if hasattr(summary_data, "__dict__"):
                    detail += " Does _get_summary_data return self or another object?"
                raise TypeError(f"_get_summary_data must return a dictionary. {detail}")
        except Exception as e:
            import traceback

            tb_str = traceback.format_exc()
            return (
                f"Error generating summary data in {self.__class__.__name__}._get_summary_data:\n"
                f"  Error Type: {type(e).__name__}\n  Error Details: {e}\n"
                f"  Traceback:\n{tb_str}"
                "Please check the implementation of _get_summary_data."
            )

        width = DEFAULT_SUMMARY_WIDTH

        all_parts: List[str] = []

        title_str = self._format_title(width)
        all_parts.append(title_str)

        general_info_data = summary_data.get("general")
        if isinstance(general_info_data, dict) and general_info_data:
            display_general = {"Model:": self.model_name or "N/A"}
            display_general.update(self._get_datetime_info())
            display_general.update({k: str(v) for k, v in general_info_data.items()})
            general_block_str = self._format_multicolumn_dict(display_general, width)
            if general_block_str.strip():
                all_parts.append(general_block_str)

        coefficients_info = summary_data.get("coefficients")
        if coefficients_info:
            coeffs_table_str = self._format_table(coefficients_info, width)
            if coeffs_table_str.strip():
                all_parts.append("Coefficients:\n" + coeffs_table_str)

        tables_dict = summary_data.get("tables")
        if isinstance(tables_dict, dict) and tables_dict:
            for name, table_info_item in tables_dict.items():
                table_item_str = self._format_table(table_info_item, width)
                if table_item_str.strip():
                    all_parts.append(f"{name}:\n{table_item_str}")

        additional_stats_data = summary_data.get("additional_stats")
        if isinstance(additional_stats_data, dict) and additional_stats_data:
            display_stats = {k: str(v) for k, v in additional_stats_data.items()}
            add_stats_block_str = self._format_multicolumn_dict(display_stats, width)
            if add_stats_block_str.strip():
                all_parts.append(add_stats_block_str)

        notes_block_str = self._format_notes(width)
        if notes_block_str.strip():
            all_parts.append(notes_block_str)

        if len(all_parts) <= 1:
            return all_parts[0].rstrip() + "\n" if all_parts else "\n"

        final_summary_content = all_parts[0]
        default_sep_line = self._format_separator(width, DEFAULT_SEPARATOR_CHAR) + "\n"
        final_sep_line = self._format_separator(width, FINAL_SEPARATOR_CHAR) + "\n"

        # Title already ends with a default separator.
        # Process blocks after title.
        # Each block string from formatters already ends with '\n'.
        for i in range(1, len(all_parts)):
            block_content_with_its_nl = all_parts[i]
            # Add default separator if previous part wasn't one and current part has content
            # The title part (all_parts[0]) ends with a default separator.
            # So, for subsequent parts, we just append them.
            # Then, if it's not the LAST content part, add a separator AFTER it.
            final_summary_content += block_content_with_its_nl
            if i < len(all_parts) - 1:  # If there's another block after this one
                final_summary_content += default_sep_line
            # No else needed here for final_sep_line, it's handled after the loop

        # Now, ensure the whole thing ends with final_sep_line
        # Remove any trailing default_sep_line if it's the last thing
        if final_summary_content.rstrip().endswith(default_sep_line.strip()):
            final_summary_content = (
                final_summary_content.rstrip()[
                    : -len(default_sep_line.strip())
                ].rstrip()
                + "\n"
            )

        final_summary_content = (
            final_summary_content.rstrip() + "\n" + final_sep_line.strip()
        )

        return final_summary_content.rstrip() + "\n"
