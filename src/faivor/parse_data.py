from dataclasses import dataclass
import json
import pandas as pd
from pathlib import Path
import csv
from typing import List, Dict, Any, Optional, Tuple

from pydantic import BaseModel

from faivor.model_metadata import ModelMetadata



@dataclass
class ColumnMetadata():
    """
    Represents metadata for  a single column in a CSV file.
    """
    id: str
    name_csv: str
    name_model: str
    categorical: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the ColumnMetadata instance to a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the column metadata.
        """
        return {
            "id": self.id,
            "name_csv": self.name_csv,
            "name_model": self.name_model,
            "categorical": self.categorical,
        }
    
    @staticmethod
    def load_from_dict(data: Dict[str, Any]) -> list["ColumnMetadata"]:
        """
        Load column metadata from a dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary containing column metadata.

        Returns
        -------
        list[ColumnMetadata]
            List of ColumnMetadata instances.
        """
        return [
            ColumnMetadata(
                id=col["id"],
                name_csv=col["name_csv"],
                name_model=col.get("name_model", col["name_csv"]),
                categorical=col.get("categorical", False)
            ) for col in data.get("columns", [])
        ]




def detect_delimiter(csv_path: Path) -> str:
    """
    Detect the delimiter used in a CSV file.

    Parameters
    ----------
    csv_path : Path
        Path to the CSV file.

    Returns
    -------
    str
        The detected delimiter (comma, semicolon, tab, or pipe).

    Raises
    ------
    IOError
        If the file cannot be opened.
    """
    candidate_delimiters = ";,\t|"
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            # A fixed byte count can truncate the sample before it contains
            # at least two complete rows, which makes csv.Sniffer() unable
            # to reliably count delimiter occurrences (e.g. wide CSVs with
            # many columns or long values). Read enough complete lines
            # instead of a fixed number of bytes, capping the total size
            # read to avoid loading an entire huge file into memory.
            lines = []
            total_bytes = 0
            max_bytes = 1024 * 100  # generous cap, ~100 lines worth for wide CSVs
            min_lines = 2
            for line in f:
                lines.append(line)
                total_bytes += len(line.encode("utf-8"))
                if len(lines) >= min_lines and total_bytes >= 1024:
                    break
                if total_bytes >= max_bytes:
                    break
            sample = "".join(lines)

        if not sample:
            raise IOError("CSV file is empty")

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=candidate_delimiters)
            return dialect.delimiter
        except csv.Error:
            # Fall back to manually counting delimiter occurrences on the
            # first line: pick whichever candidate delimiter appears most
            # often, as long as it appears at least once.
            first_line = lines[0] if lines else sample
            counts = {d: first_line.count(d) for d in candidate_delimiters}
            best_delim, best_count = max(counts.items(), key=lambda kv: kv[1])
            if best_count > 0:
                return best_delim
            raise
    except (FileNotFoundError, IOError) as e:
        raise IOError(f"Could not open CSV file: {e}") from e

def detect_decimal(csv_path: Path) -> str:
    """
    Detect the decimal separator used in a CSV file.

    Parameters
    ----------
    csv_path : Path
        Path to the CSV file.

    Returns
    -------
    str
        The detected decimal separator (comma or dot).

    Raises
    ------
    IOError
        If the file cannot be opened.
    """
    try:
        delimiter = detect_delimiter(csv_path)
        with open(csv_path, "r", encoding="utf-8") as f:
            # Skip header
            header = f.readline()
            # Read the second line (first data row)
            line = f.readline()
            if not line:
                raise ValueError("CSV file does not contain data rows to detect decimal separator")
            fields = line.strip().split(delimiter)
            # Assume dot as decimal unless a comma is found in a value
            for field in fields:
                if "," in field and "." not in field:
                    return ","
            return "."
    except (FileNotFoundError, IOError) as e:
        raise IOError(f"Could not open CSV file: {e}") from e

def load_csv(csv_path: Path) -> tuple[pd.DataFrame, List[str]]:
    """
    Load a CSV file into a DataFrame using the detected delimiter.

    Parameters
    ----------
    csv_path : Path
        Path to the CSV file.

    Returns
    -------
    tuple[pd.DataFrame, List[str]]
        - df: the loaded DataFrame
        - columns: list of all column names in the CSV

    Raises
    ------
    pd.errors.ParserError
        If the CSV cannot be parsed.
    """
    delimiter = detect_delimiter(csv_path)
    decimal = detect_decimal(csv_path)
    df = pd.read_csv(csv_path, sep=delimiter, decimal=decimal)
    return df, df.columns.tolist()


def validate_dataframe_format(
    metadata: ModelMetadata,
    csv_df: pd.DataFrame
) -> tuple[bool, str | None]:
    """
    Validate that all required input and output columns exist in the CSV.

    Parameters
    ----------
    metadata : ModelMetadata
        Object with `.inputs` (list of input_label) and `.output` (str).
    csv_df : pd.DataFrame
        DataFrame containing the CSV data.

    Returns
    -------
    tuple[bool, str | None]
        (True, None) if all required columns are present,
        (False, message) otherwise.
    """

    required = [inp.input_label for inp in metadata.inputs] + [metadata.output]
    missing = [col for col in required if col not in csv_df.columns]

    if missing:
        return False, f"Missing required columns: {', '.join(missing)}"

    return True, None


def create_json_payloads(
    metadata: ModelMetadata,
    csv_path: Path,
    column_metadata: list[ColumnMetadata],
    complete_cases: bool = True
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Create JSON payloads for input and output data based on provided metadata and CSV file.

    This function reuses `validate_dataframe_format` to ensure the CSV is valid before extracting payloads.

    Parameters
    ----------
    metadata : ModelMetadata
        Parsed model metadata with `.inputs` (list of {"input_label": ...})
        and `.output` (str).
    csv_path : Path
        Path to the CSV file.
    column_metadata : list[ColumnMetadata]
        Additional metadata for each column, used to rename columns in the DataFrame.

    Returns
    -------
    Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]
        - inputs: list of dicts for model inputs
        - outputs: list of dicts for model outputs

    Raises
    ------
    ValueError
        If the CSV is missing required columns (propagated from `validate_csv_format`).
    """

    df, all_columns = load_csv(csv_path)

    if column_metadata:
        # If data metadata is provided, rename columns in the DataFrame that mates csv_name to model_name
        for col in column_metadata:
            if col.name_csv in df.columns:
                df.rename(columns={col.name_csv: col.name_model}, inplace=True)
                
    is_valid, message = validate_dataframe_format(metadata, df)

    if not is_valid:
        raise ValueError(f"CSV format validation failed: {message}")

    input_cols = [inp.input_label for inp in metadata.inputs]
    output_col = metadata.output

    if complete_cases:
        # Make a subset of complete cases for inputs and outputs
        df = df.dropna(subset=input_cols + [output_col])

    inputs = df[input_cols].to_dict(orient="records")
    outputs = df[[output_col]].to_dict(orient="records")

    return inputs, outputs