import json
import os
import sys
from pathlib import Path
from faivor.model_metadata import ModelMetadata
from faivor.parse_data import ColumnMetadata, create_json_payloads
from faivor.run_docker import execute_model

def run():
    model_name = "pilot-model_1"
    model_dir = Path("tests/data/models") / model_name
    
    metadata_json = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    model_metadata = ModelMetadata(metadata_json)
    
    csv_path = model_dir / "data.csv"
    column_metadata_json = json.loads((model_dir / "column_metadata.json").read_text())
    column_metadata = ColumnMetadata.load_from_dict(column_metadata_json)
    
    inputs, _ = create_json_payloads(model_metadata, csv_path, column_metadata)
    total_rows = len(inputs)
    print(f"Total input rows: {total_rows}")
    
    for i in range(1, total_rows + 1):
        prefix = inputs[:i]
        try:
            execute_model(model_metadata, prefix)
        except Exception as e:
            error_msg = str(e)
            stderr = getattr(e, 'stderr', b'').decode() if hasattr(e, 'stderr') and isinstance(e.stderr, bytes) else str(getattr(e, 'stderr', ''))
            full_error = error_msg + " " + stderr
            
            # The prompt asks for the exception message and details if it fails.
            # We print them and exit.
            print(f"First failing prefix index: {i}")
            print(f"Exception message: {full_error.strip()}")
            print(f"Exact input row at index {i}: {inputs[i-1]}")
            return

    print("No prefix failed.")

if __name__ == "__main__":
    run()
