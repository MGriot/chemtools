# -*- coding: utf-8 -*-
"""
This module provides base model functionality for statistical analysis tools.
It contains the BaseModel abstract base class which implements common model operations
and defines the interface for specific model implementations.
"""

import pickle
import datetime
import math
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional


# --- Constants for Formatting ---
DEFAULT_SUMMARY_WIDTH: int = 90
DEFAULT_SEPARATOR_CHAR: str = "-"
FINAL_SEPARATOR_CHAR: str = "="
DEFAULT_COLUMN_SEPARATOR: str = "  "
DEFAULT_NOTES_INDENT: str = "    "
MIN_KEY_VALUE_DISPLAY_WIDTH: int = 10 # Minimum width for key/value display parts
MULTI_COL_PADDING: int = 4 # Approx padding/spaces around key/value in multicolumn

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

        This method must be implemented by concrete model classes. It should
        return a dictionary containing the data needed to build the summary
        report.

        Expected Structure:
        {
            "general": Dict[str, str],  # Key-value pairs for general info
            "coefficients": List[List[Any]], # Header row + data rows
            "tables": Dict[str, List[List[Any]]], # Named tables (header + data)
            "additional_stats": Dict[str, str] # Key-value pairs
        }
        Keys are optional; if a key is missing, that section won't be printed.

        Returns:
            A dictionary containing data structured for the summary.
        """
        return {}

    def _create_general_summary(
        self, n_variables: int, n_objects: int, **kwargs
    ) -> Dict[str, Dict[str, str]]:
        """Creates the 'general' part of the summary data dictionary.

        Args:
            n_variables: Number of variables in the model.
            n_objects: Number of objects/observations used.
            **kwargs: Additional key-value pairs to include in the general section.
                      Values will be converted to strings.

        Returns:
            A dictionary containing the 'general' key mapped to the summary data.
        """
        summary_data = {
            "general": {
                "No. Variables": str(n_variables),
                "No. Objects": str(n_objects),
            },
        }
        # Ensure kwargs values are strings for consistent formatting
        summary_data["general"].update({k: str(v) for k, v in kwargs.items()})
        return summary_data

    # --- Summary Formatting Helpers ---

    def _format_separator(self, width: int, char: str = DEFAULT_SEPARATOR_CHAR) -> str:
        """Creates a separator line string."""
        return char * width

    def _format_title(self, width: int) -> str:
        """Formats the main title section of the summary."""
        title = f"{self.method or self.model_name} Summary"
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

        Returns:
            Tuple[int, int, int]: (width_per_column, left_column_width, key_display_width)
        """
        if not data or num_cols <= 0:
            return 0, 0, 0

        col_width = total_width // num_cols
        # Approximate original asymmetric split using integer division
        left_col_width = total_width // 2
        #right_col_width = total_width - left_col_width

        # Estimate max key length dynamically, add buffer for ': ' etc.
        try:
            # Consider both key and value lengths, add buffer for spacing
            max_len = max(max(len(k), len(str(v))) for k, v in data.items()) + MULTI_COL_PADDING
        except ValueError: # Handle empty data dictionary
            max_len = MIN_KEY_VALUE_DISPLAY_WIDTH # Default minimum

        # Ensure key display width doesn't exceed available space per item column,
        # minus space for value and separators, but enforce a minimum.
        key_col_display_width = max(
            MIN_KEY_VALUE_DISPLAY_WIDTH,
            min(max_len, col_width - MIN_KEY_VALUE_DISPLAY_WIDTH) # Ensure some space left for value
        )

        # Return total width per column, width of the first column block, and calculated key display width
        return col_width, left_col_width, key_col_display_width


    def _format_multicolumn_dict(
        self, data: Dict[str, str], total_width: int, num_cols: int = 2
    ) -> str:
        """Formats a dictionary into a multi-column string (e.g., for general stats)."""
        if not data or num_cols <= 0:
            return ""

        _col_width, left_col_width, key_disp_width = self._calculate_multicolumn_widths(
            data, total_width, num_cols
        )
        right_col_width = total_width - left_col_width

        items = list(data.items())
        num_rows = math.ceil(len(items) / num_cols)
        lines: List[str] = []

        for row_idx in range(num_rows):
            row_parts: List[str] = []
            for col_idx in range(num_cols):
                item_idx = row_idx + col_idx * num_rows
                if item_idx < len(items):
                    key, value = items[item_idx]
                    value_str = str(value) # Ensure value is string

                    if col_idx == 0: # Left column formatting
                        # Calculate available width for value in the left half
                        value_width = left_col_width - key_disp_width - len(DEFAULT_COLUMN_SEPARATOR)
                        part = (f"{str(key):<{key_disp_width}}"
                                f"{DEFAULT_COLUMN_SEPARATOR}"
                                f"{value_str:>{value_width}}")
                        # Pad the entire left part to its designated width
                        row_parts.append(part.ljust(left_col_width))
                    else: # Right column formatting
                        # Calculate available width for value in the right half
                        # Account for the separator between key/value AND the separator from the left col
                        value_width = right_col_width - key_disp_width - (len(DEFAULT_COLUMN_SEPARATOR)*2)
                        part = (f"{DEFAULT_COLUMN_SEPARATOR}" # Separator from left col
                                f"{str(key):<{key_disp_width}}"
                                f"{DEFAULT_COLUMN_SEPARATOR}" # Separator key/value
                                f"{value_str:>{value_width}}")
                        # Pad the entire right part to its designated width
                        row_parts.append(part.ljust(right_col_width))
                else:
                    # Pad empty slots to maintain column structure
                    part_width = left_col_width if col_idx == 0 else right_col_width
                    row_parts.append(" " * part_width)

            # Join columns for the row, strip trailing spaces only from the very end of the line
            lines.append("".join(row_parts).rstrip())

        return "\n".join(lines) + "\n"


    def _calculate_table_col_widths(self, headers: List[Any], total_width: int) -> List[int]:
        """Calculates widths for table columns based on headers and total width."""
        num_cols = len(headers)
        if num_cols == 0:
            return []

        # Account for space needed for separators between columns
        separator_space = len(DEFAULT_COLUMN_SEPARATOR) * (num_cols - 1)
        available_width = total_width - separator_space

        if available_width <= 0: # Avoid division issues if width too small
            # Fallback: assign minimal width if possible, else 0
            base_width = 1 if num_cols <= total_width else 0
            return [base_width] * num_cols


        base_col_width = available_width // num_cols
        remainder = available_width % num_cols

        col_widths = [base_col_width] * num_cols
        # Distribute remainder width to the first few columns for slightly wider appearance
        for i in range(remainder):
            col_widths[i] += 1

        return col_widths

    def _format_table(self, table_data: List[List[Any]], total_width: int) -> str:
        """Formats a list of lists (table) into a fixed-width string."""
        if not table_data or not table_data[0]:
            # Handle empty table or table with no header/columns
            return ""

        headers = table_data[0]
        col_widths = self._calculate_table_col_widths(headers, total_width)
        num_cols = len(headers)

        lines: List[str] = []
        for row in table_data:
            # Ensure row has correct number of elements (pad with empty strings if short)
            str_row = [str(cell) for cell in row[:num_cols]]
            padded_row = str_row + [""] * (num_cols - len(str_row))

            formatted_cells = [
                f"{cell:<{col_widths[j]}}" for j, cell in enumerate(padded_row)
            ]
            # Join cells with separator, strip trailing space from the final line content
            lines.append(DEFAULT_COLUMN_SEPARATOR.join(formatted_cells).rstrip())

        return "\n".join(lines) + "\n"


    def _wrap_text(self, text: str, width: int) -> List[str]:
        """Wraps text to a given width, handling line breaks and long words.

        Args:
            text: The input text string.
            width: The maximum width for each line.

        Returns:
            A list of strings, where each string is a wrapped line.
        """
        if width <= 0:
            return [text] # Avoid infinite loops or errors

        lines: List[str] = []
        # Preserve existing line breaks in the input text
        paragraphs = text.splitlines()

        for paragraph in paragraphs:
            if not paragraph.strip(): # Handle empty lines or lines with only whitespace
                lines.append("")
                continue

            words = paragraph.split()
            if not words: # Handle paragraphs that become empty after splitting
                lines.append("")
                continue

            current_line = ""
            for word in words:
                potential_line = f"{current_line} {word}".strip() if current_line else word

                if len(potential_line) <= width:
                    current_line = potential_line
                else:
                    # Word doesn't fit on the current line
                    if current_line: # Add the completed line first
                        lines.append(current_line)

                    # Handle long word that exceeds width by itself
                    if len(word) > width:
                        # Break the long word across multiple lines
                        start = 0
                        while start < len(word):
                            lines.append(word[start:start + width])
                            start += width
                        current_line = "" # Word fully processed, start fresh
                    else:
                        # Start new line with the current word
                        current_line = word

            if current_line: # Add the last constructed line of the paragraph
                lines.append(current_line)

        # Ensure single empty string if input was effectively empty
        if not lines and not text.strip():
            return [""]
        elif not lines and paragraphs and not any(p.strip() for p in paragraphs):
            return [""] * len(paragraphs) # Preserve number of empty lines if that was the input


        return lines


    def _format_notes(self, total_width: int) -> str:
        """Formats the notes section with numbering and text wrapping."""
        if not self.notes:
            return ""

        lines: List[str] = ["Notes:"]
        # Calculate width available for the note text itself, accounting for "[i] " indent
        note_text_width = total_width - len(DEFAULT_NOTES_INDENT)

        for i, note_text in enumerate(self.notes, 1):
            # Wrap the text for the current note
            wrapped_lines = self._wrap_text(note_text, note_text_width)
            for j, line in enumerate(wrapped_lines):
                # Add prefix: "[i] " for the first line, indent for subsequent lines
                prefix = f"[{i}] ".ljust(len(DEFAULT_NOTES_INDENT)) if j == 0 else DEFAULT_NOTES_INDENT
                lines.append(f"{prefix}{line}")

        return "\n".join(lines) + "\n"

    # --- Main Summary Property ---

    @property
    def summary(self) -> str:
        """
        Generates a formatted summary string of the model results.

        Constructs the summary by fetching data via `_get_summary_data` and
        formatting each section (general, coefficients, tables, stats, notes)
        using helper methods.
        """
        try:
            summary_data = self._get_summary_data()
        except (KeyError, ValueError, TypeError) as e:
            # Handle potential errors in data generation from subclasses
            return (f"Error generating summary data: {e}\n"
                    f"Please check the implementation of _get_summary_data in {self.__class__.__name__}.")


        width = DEFAULT_SUMMARY_WIDTH
        summary_parts: List[str] = [] # Use a list to build summary parts efficiently

        # 1. Header Title Section
        summary_parts.append(self._format_title(width))

        # 2. General Model Information Section
        general_info = summary_data.get("general")
        if isinstance(general_info, dict) and general_info:
            # Prepare data: Add model name, date/time first, then user data
            # Use copy to avoid modifying original data potentially returned by getter
            display_general = {"Model:": self.model_name}
            display_general.update(self._get_datetime_info())
            # Ensure values are strings
            display_general.update({k: str(v) for k, v in general_info.items()})

            summary_parts.append(self._format_multicolumn_dict(display_general, width))
            summary_parts.append(self._format_separator(width) + "\n")
        elif general_info is not None:
            # Log or warn if 'general' exists but isn't a non-empty dict?
            pass


        # 3. Coefficients Section
        coefficients = summary_data.get("coefficients")
        if isinstance(coefficients, list) and coefficients:
            summary_parts.append("Coefficients:\n")
            summary_parts.append(self._format_table(coefficients, width))
            summary_parts.append(self._format_separator(width) + "\n")

        # 4. Additional Tables Section
        tables = summary_data.get("tables")
        if isinstance(tables, dict) and tables:
            # Add extra newline if coefficients were printed, for spacing
            if coefficients:
                summary_parts.append("\n")

            for name, data in tables.items():
                if isinstance(data, list) and data:
                    summary_parts.append(f"{name}:\n")
                    summary_parts.append(self._format_table(data, width))
                    summary_parts.append(self._format_separator(width) + "\n")

        # 5. Additional Statistics Section
        additional_stats = summary_data.get("additional_stats")
        if isinstance(additional_stats, dict) and additional_stats:
            # Ensure values are strings
            display_stats = {k: str(v) for k, v in additional_stats.items()}

            # Add spacing if previous sections exist
            if coefficients or tables:
                # Check if last part already ends with double newline potentially
                if not summary_parts[-1].endswith("\n\n"):
                    # Check if last part is a separator line, if so, add newline before stats
                    if summary_parts[-1].strip() == self._format_separator(width):
                        summary_parts.append("\n")
                    # Otherwise, ensure a newline separates from previous content
                    elif not summary_parts[-1].endswith("\n"):
                        summary_parts[-1] += "\n"


            summary_parts.append(self._format_multicolumn_dict(display_stats, width))
            # This section should end with the FINAL separator
            summary_parts.append(self._format_separator(width, FINAL_SEPARATOR_CHAR) + "\n")


        # 6. Final Separator Logic (if 'additional_stats' was absent)
        if not (isinstance(additional_stats, dict) and additional_stats):
            # Check if there was content after the title
            if len(summary_parts) > 1:
                # Remove the last default separator if it exists
                if summary_parts[-1] == self._format_separator(width) + "\n":
                    summary_parts.pop()
                # Add the final '=' separator if content exists
                if summary_parts[-1] != self._format_title(width): # Avoid adding if only title exists
                    summary_parts.append(self._format_separator(width, FINAL_SEPARATOR_CHAR) + "\n")


        # 7. Notes Section
        notes_str = self._format_notes(width)
        if notes_str:
            # Check if the last element is the final separator and replace it
            if summary_parts and summary_parts[-1] == self._format_separator(width, FINAL_SEPARATOR_CHAR) + "\n":
                summary_parts[-1] = self._format_separator(width, DEFAULT_SEPARATOR_CHAR) + "\n"
            # Ensure newline separation if the last part wasn't a separator
            elif summary_parts and not summary_parts[-1].endswith("\n"):
                summary_parts[-1] += "\n"

            summary_parts.append(notes_str)
            # Notes section should end with a default separator
            summary_parts.append(self._format_separator(width, DEFAULT_SEPARATOR_CHAR) + "\n")


        # --- Final Assembly ---
        final_summary = "".join(summary_parts).rstrip() # Join all parts and remove trailing whitespace

        # Ensure the entire output ends correctly
        default_sep_ending = self._format_separator(width, DEFAULT_SEPARATOR_CHAR)
        final_sep_ending = self._format_separator(width, FINAL_SEPARATOR_CHAR)

        if final_summary.endswith(default_sep_ending):
            # Replace trailing default separator with final one
            final_summary = final_summary[:-len(default_sep_ending)] + final_sep_ending
        elif not final_summary.endswith(final_sep_ending) and final_summary:
            # Add final separator if missing (and summary is not empty)
            # Check if it only contains the title block ending in default sep
            title_block = self._format_title(width).rstrip()
            if final_summary == title_block:
                final_summary = final_summary[:-len(default_sep_ending)] + final_sep_ending
            else:
                final_summary += "\n" + final_sep_ending
        elif not final_summary: # Handle case where summary_data was empty
            # Return just the title if model/method names exist
            title_str = self._format_title(width).rstrip()
            # Adjust final separator
            if title_str.endswith(default_sep_ending):
                title_str = title_str[:-len(default_sep_ending)] + final_sep_ending
            final_summary = title_str


        return final_summary + "\n" # Add final newline for standard output
