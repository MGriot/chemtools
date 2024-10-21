import pickle
import datetime
import math
from abc import ABC, abstractmethod


class BaseModel(ABC):  # Make BaseModel an Abstract Base Class
    """
    Abstract base class for statistical models, providing common
    functionalities like saving/loading and a basic summary.
    """

    def __init__(self):
        self.model_name = "Base Model"  # Default name
        self.method = None  # Default method
        self.notes = []  # Initialize an empty list to store notes

    def save(self, filename):
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as file:
            return pickle.load(file)

    @abstractmethod
    def _get_summary_data(self):
        """
        Abstract method that should be implemented by subclasses
        to return a dictionary of data for the summary.
        Should return a dictionary where keys are the names of the parameters
        and values are the values of the parameters.
        """
        pass

    @property
    def summary(self):
        """
        Returns a formatted summary of the model, potentially
        splitting into multiple tables. The general statistics table
        can be organized into multiple columns.
        """
        summary_data = self._get_summary_data()
        a = 45
        b = a * 2
        now = datetime.datetime.now()

        summary_string = "=" * b + "\n"
        summary_string += f"{self.method} Summary".center(b) + "\n"
        summary_string += "-" * b + "\n"

        # General Model Statistics (organized into columns)
        if "general" in summary_data:
            general_stats = summary_data["general"]
            # ------------------------------------------------
            max_key_length = (
                max(
                    max(len(str(key)), len(str(value)))
                    for key, value in general_stats.items()
                )
                + 2
            )

            # --- Add top-level information ---
            general_stats["Model:"] = self.model_name
            general_stats["Date:"] = now.strftime("%a, %d %b %Y")
            general_stats["Time:"] = now.strftime("%H:%M:%S")

            # --- Format the remaining general stats ---
            num_cols = 2  # You can adjust the number of columns here
            col_width = a  # Adjust column width as needed
            # Define the order of keys for display
            # Define the order of keys for display, forcing Model, Date, and Time to be first
            key_order = ["Model:", "Date:", "Time:"] + [
                key
                for key in general_stats.keys()
                if key not in ["Model:", "Date:", "Time:"]
            ]
            # Calculate the number of rows needed
            num_rows = math.ceil(len(key_order) / num_cols)

            # Iterate through rows and columns
            for row in range(num_rows):
                row_string = ""
                for col in range(num_cols):
                    index = row + col * num_rows
                    if index < len(key_order):
                        key = key_order[index]
                        value = general_stats[key]
                        # Align value text to the right with spacing
                        if col == 0:
                            row_string += f"{str(key):<{max_key_length-4}}  {value:>{col_width - max_key_length}}  "
                        else:
                            row_string += f"  {str(key):<{max_key_length}}  {value:>{col_width - max_key_length-4}}"
                summary_string += row_string + "\n"

            summary_string += "-" * b + "\n"

        # Coefficient Table
        if "coefficients" in summary_data:
            summary_string += "\nCoefficients:\n"
            summary_string += "-" * b + "\n"
            coef_table = summary_data["coefficients"]

            # Calculate column widths based on the full width 'b'
            num_cols = len(coef_table[0])
            col_width = b // num_cols  # Integer division to get equal widths

            for row in coef_table:
                summary_string += (
                    "  ".join(f"{str(value):<{col_width}}" for value in row) + "\n"
                )
            summary_string += "-" * b
            # Additional Tables Section
        if "tables" in summary_data:
            for table_name, table_data in summary_data["tables"].items():
                summary_string += f"\n{table_name}:\n"
                summary_string += "-" * b + "\n"

                # Calculate column widths for this table
                num_cols = len(table_data[0])
                col_width = b // num_cols

                for row in table_data:
                    summary_string += (
                        "  ".join(f"{str(value):<{col_width}}" for value in row) + "\n"
                    )
                summary_string += "-" * b + "\n"

            # --- Add new section for additional statistics ---
        if "additional_stats" in summary_data:
            summary_string += "\n"
            additional_stats = summary_data["additional_stats"]

            # --- Format the additional stats into two columns ---
            num_cols = 2
            col_width = a
            max_key_length = max(len(key) for key in additional_stats) + 2
            num_rows = math.ceil(len(additional_stats) / num_cols)

            keys = list(additional_stats.keys())  # Get keys as a list

            for row in range(num_rows):
                row_string = ""
                for col in range(num_cols):
                    index = row + col * num_rows
                    if index < len(keys):
                        key = keys[index]
                        value = additional_stats[key]
                        if col == 0:
                            row_string += f"{key:<{max_key_length - 4}}  {value:>{col_width - max_key_length}}  "
                        else:
                            row_string += f"  {key:<{max_key_length}}  {value:>{col_width - max_key_length-4}}"
                summary_string += row_string + "\n"

            summary_string += "=" * b

        # --- Add notes section ---
        if self.notes:
            summary_string += "\nNotes:\n"
            for i, note in enumerate(self.notes, 1):
                note_lines = self._wrap_text(note, b - 4)  # b - 4 for "[i] " and space
                for j, line in enumerate(note_lines):
                    if j == 0:
                        summary_string += f"[{i}] {line}\n"
                    else:
                        summary_string += f"    {line}\n"
            summary_string += "-" * b  # Add a separator line after notes

        return summary_string

    def _wrap_text(self, text, width):
        """Wraps text to a given width."""
        words = text.split()
        lines = []
        current_line = ""
        for word in words:
            if len(current_line) + len(word) <= width:
                current_line += word + " "
            else:
                lines.append(current_line.strip())
                current_line = word + " "
        lines.append(current_line.strip())  # Add the last line
        return lines
