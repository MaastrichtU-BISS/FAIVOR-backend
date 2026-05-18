import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional
from importlib.metadata import version as get_version

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# pylint: disable=no-name-in-module
import pandas as pd
from pydantic import BaseModel

from faivor.calculate_metrics import MetricsCalculator
from faivor.metrics.classification import classification_metrics
from faivor.metrics.regression import regression_metrics
from faivor.model_metadata import ModelMetadata
from faivor.parse_data import ColumnMetadata, create_json_payloads, load_csv, validate_dataframe_format
from faivor.run_docker import execute_model

# Get version from package metadata
try:
    __version__ = get_version("FAIVOR")
except Exception:
    __version__ = "0.1.0"  # Fallback version

app = FastAPI(
    title="FAIVOR ML Validator",
    version=__version__,
    description="FAIRmodels validator API"
)

# Add CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def create_error_response(
    status_code: int,
    error_code: str,
    message: str,
    technical_details: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> HTTPException:
    """Create a structured error response with consistent format"""
    error_detail = ErrorDetail(
        code=error_code,
        message=message,
        technical_details=technical_details,
        metadata=metadata or {}
    )
    
    return HTTPException(
        status_code=status_code,
        detail=error_detail.model_dump()
    )


@app.get("/")
async def root():
    return {"message": "Welcome to FAIVOR ML Validator", "version": __version__}


@app.get("/version")
async def get_api_version():
    """Get the API version information."""
    return {
        "name": "FAIVOR ML Validator",
        "version": __version__
    }

class ListColumnMetadataModel(BaseModel):
    columns: list[ColumnMetadata]

class ValidationResponse(BaseModel):
    valid: bool
    message: str | None = None
    csv_columns: list[str]
    model_input_columns: list[str]


class ErrorDetail(BaseModel):
    """Structured error response for better error handling in clients"""
    code: str
    message: str
    technical_details: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """Standard error response wrapper"""
    detail: ErrorDetail


@app.post(
    "/validate-csv/",
    response_model=ValidationResponse,
    summary="Validate CSV against model metadata",
    description=(
        "Uploads a FAIR model metadata JSON string and a CSV file, "
        "verifies that all required columns are present, "
        "and returns the list of CSV column names."
    ),
    responses={
        400: {
            "description": "Invalid metadata JSON or CSV format",
            "content": {
                "application/json": {
                    "example": {"message": "Invalid metadata JSON: ..."}
                }
            },
        },
    },
)
async def validate_csv(
    model_metadata: str = Form(
        ...,
        description="FAIR model metadata JSON, containing `inputs` (list of `{input_label}`) and `output` field",
    ),
    csv_file: UploadFile = File(
        ..., description="CSV file to validate; delimiter is auto-detected"
    ),
    column_metadata: ListColumnMetadataModel | None = Form(
        None,
        description="Metadata JSON, containing naming mapping for the CSV columns, as well as information about whether the column is categorical (it can be used for threshold metrics calculation) or not.",
    ),
) -> JSONResponse:
    """
    Validate a CSV file against provided metadata and return all CSV columns.
    """
    try:
        md = json.loads(model_metadata)
        metadata = ModelMetadata(md)
    except json.JSONDecodeError as e:
        raise create_error_response(
            status_code=400,
            error_code="INVALID_METADATA_JSON",
            message="Invalid metadata JSON format",
            technical_details=str(e),
            metadata={"position": e.pos if hasattr(e, 'pos') else None}
        ) from e
    except Exception as e:
        raise create_error_response(
            status_code=400,
            error_code="METADATA_PARSE_ERROR",
            message="Failed to parse model metadata",
            technical_details=str(e)
        ) from e

    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            shutil.copyfileobj(csv_file.file, tmp)
            tmp_path = Path(tmp.name)

        df_data, columns = load_csv(tmp_path)
    except pd.errors.ParserError as e:
        raise create_error_response(
            status_code=400,
            error_code="INVALID_CSV_FORMAT",
            message="Invalid CSV file format",
            technical_details=str(e),
            metadata={"file_name": csv_file.filename}
        ) from e
    except Exception as e:
        raise create_error_response(
            status_code=400,
            error_code="CSV_READ_ERROR",
            message="Failed to read CSV file",
            technical_details=str(e),
            metadata={"file_name": csv_file.filename}
        ) from e
    
    if column_metadata:
        for col in column_metadata:
            if col.name_csv in df_data.columns:
                df_data.rename(columns={col.name_csv: col.name_model}, inplace=True)

    validation_result, msg = validate_dataframe_format(metadata, df_data)
    
    # If validation failed due to missing columns, provide structured error info
    if not validation_result and msg and "Missing required columns" in msg:
        # Extract missing columns from the message
        import re
        missing_match = re.search(r"Missing required columns: (.+)", msg)
        missing_columns = []
        if missing_match:
            missing_columns = [col.strip() for col in missing_match.group(1).split(',')]
        
        # Return validation response with detailed error info
        return JSONResponse(
            content={
                "valid": False,
                "message": msg,
                "csv_columns": columns,
                "model_input_columns": [
                    input_obj.input_label for input_obj in metadata.inputs
                ],
                "error_details": {
                    "code": "MISSING_REQUIRED_COLUMNS",
                    "missing_columns": missing_columns,
                    "available_columns": columns
                }
            },
            status_code=200  # Return 200 as this is a validation result, not an error
        )

    return JSONResponse(
        content={
            "valid": validation_result,
            "message": msg,
            "csv_columns": columns,
            "model_input_columns": [
                input_obj.input_label for input_obj in metadata.inputs
            ],
        }
    )


class MetricDescription(BaseModel):
    name: str
    description: str
    type: str


@app.post(
    "/retrieve-metrics",
    response_model=list[MetricDescription],
    summary="Retrieve applicable metrics for the model",
    description=(
        "Returns a list of MetricDescription objects containing the names and descriptions "
        "of metrics applicable to the model based on the provided FAIR model metadata. "
        "Optionally filter by category (`performance`, `fairness`, `explainability`)."
    ),
)
async def retrieve_metrics(
    model_metadata: str = Form(
        ...,
        description="FAIR model metadata JSON, containing `inputs` (list of `{input_label}`) and `output` field",
    ),
) -> list[MetricDescription]:
    """
    Retrieve applicable metrics for the model.
    """
    try:
        # Parse the model metadata
        md = json.loads(model_metadata)
        metadata = ModelMetadata(md)
        model_type = metadata.metadata.get("model_type", "regression").lower()

        # detect appropriate metrics module based on model type
        metrics_module = (classification_metrics if model_type.lower() == "classification" 
                        else regression_metrics)

        # Collect metrics
        performance_metrics = [
                MetricDescription(
                    name=metric.regular_name, description=metric.description, type = "performance"
                )
                for metric in metrics_module.PERFORMANCE_METRICS
            ]
        fairness_metrics = [
                MetricDescription(
                    name=metric.regular_name, description=metric.description, type = "fairness"
                )
                for metric in metrics_module.FAIRNESS_METRICS
            ]
        explainability_metrics= [
                MetricDescription(
                    name=metric.regular_name, description=metric.description, type = "explainability"
                )
                for metric in metrics_module.EXPLAINABILITY_METRICS
            ]


        # Return all metrics as a flat list
        return performance_metrics + fairness_metrics + explainability_metrics

    except json.JSONDecodeError as e:
        raise create_error_response(
            status_code=400,
            error_code="INVALID_METADATA_JSON",
            message="Invalid metadata JSON format",
            technical_details=str(e),
            metadata={"position": e.pos if hasattr(e, 'pos') else None}
        ) from e
    except Exception as e:
        raise create_error_response(
            status_code=500,
            error_code="METRICS_RETRIEVAL_ERROR",
            message="Failed to retrieve metrics information",
            technical_details=str(e)
        ) from e

    
class ModelMetrics(BaseModel):
    model_name: str
    metrics: dict[str, float]



@app.post(
    "/validate-model",
    summary="Validate ML model",
    description=(
        "Validates the ML model by checking the provided FAIR model metadata, "
        "the CSV file, and the column metadata. "
        "The `column_metadata` input must follow the specified format."
    ),
)
async def validate_model(
    model_metadata: str = Form(
        ...,
        description="FAIR model metadata JSON, containing `inputs` (list of `{input_label}`) and `output` field",
    ),
    csv_file: UploadFile = File(
        ..., description="CSV file to validate; delimiter is auto-detected"
    ),
    column_metadata: ListColumnMetadataModel | None = Form(
        None,
        description="Metadata JSON, containing naming mapping for the CSV columns, as well as information about whether the column is categorical (it can be used for threshold metrics calculation) or not. ",
    ),
) -> JSONResponse:
    """
    Validate a model with metadata and CSV data.
    """
    try:
        # Parse the model metadata
        md = json.loads(model_metadata)
        metadata = ModelMetadata(md)
        model_name = metadata.model_name or "unknown_model"

        # Parse column metadata if provided
        try:
            if column_metadata:
                col_metadata = json.loads(column_metadata) if column_metadata else {}
                columns_metadata : list[ColumnMetadata] = ColumnMetadata.load_from_dict(col_metadata)
            else:
                columns_metadata = []
        except json.JSONDecodeError as exc:
            raise create_error_response(
                status_code=400,
                error_code="INVALID_COLUMN_METADATA_JSON",
                message="Invalid column metadata JSON format",
                technical_details=str(exc)
            ) from exc

        # Load CSV data
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            shutil.copyfileobj(csv_file.file, tmp)
            tmp_path = Path(tmp.name)


        # Prepare inputs and expected outputs
        inputs, expected_outputs = create_json_payloads(metadata, tmp_path, columns_metadata)

        # Execute model and get predictions
        try:
            execution_result = execute_model(metadata, inputs)
            predictions = execution_result["predictions"]
            docker_image_sha256 = execution_result.get("docker_image_sha256")
        except ValueError as e:
            # ValueError indicates data validation errors (invalid/out-of-range values)
            # This is a client error (400), not a server error
            error_msg = str(e)
            raise create_error_response(
                status_code=400,
                error_code="DATA_VALIDATION_ERROR",
                message="The model rejected the input data due to validation errors",
                technical_details=error_msg,
                metadata={
                    "model_name": model_name,
                    "error_type": "data_validation",
                    "user_guidance": "Check that your data values are within the expected ranges for this model. "
                                   "Review the model documentation for valid input constraints."
                }
            ) from e
        except RuntimeError as e:
            # Handle specific Docker/container errors
            error_msg = str(e)
            if "timeout" in error_msg.lower():
                raise create_error_response(
                    status_code=503,
                    error_code="MODEL_EXECUTION_TIMEOUT",
                    message="Model execution timed out",
                    technical_details=error_msg,
                    metadata={"model_name": model_name, "timeout": "300s"}
                ) from e
            elif "status code 4" in error_msg.lower() or "prediction failed" in error_msg.lower():
                # Model execution failed (status 4) - likely a data processing error within the model
                raise create_error_response(
                    status_code=400,
                    error_code="MODEL_PROCESSING_ERROR",
                    message="The model failed to process the input data",
                    technical_details=error_msg,
                    metadata={
                        "model_name": model_name,
                        "error_type": "model_processing",
                        "user_guidance": "The model encountered an error while processing your data. "
                                       "This may be due to invalid values, missing data, or data format issues. "
                                       "Check the technical details below for more information."
                    }
                ) from e
            elif "docker" in error_msg.lower() or "container" in error_msg.lower():
                raise create_error_response(
                    status_code=503,
                    error_code="CONTAINER_EXECUTION_ERROR",
                    message="Failed to execute model in Docker container",
                    technical_details=error_msg,
                    metadata={"model_name": model_name}
                ) from e
            else:
                raise create_error_response(
                    status_code=500,
                    error_code="MODEL_EXECUTION_FAILED",
                    message="Model execution failed",
                    technical_details=error_msg,
                    metadata={"model_name": model_name}
                ) from e
        except Exception as e:
            raise create_error_response(
                status_code=500,
                error_code="MODEL_EXECUTION_ERROR",
                message="Unexpected error during model execution",
                technical_details=str(e),
                metadata={"model_name": model_name}
            ) from e

        # Initialize MetricsCalculator
        try:
            metrics_calculator = MetricsCalculator(
                model_metadata=metadata,
                predictions=predictions,
                expected_outputs=expected_outputs,
                inputs=inputs,
            )
        except Exception as e:
            raise create_error_response(
                status_code=500,
                error_code="METRICS_CALCULATOR_INIT_ERROR",
                message="Failed to initialize metrics calculator",
                technical_details=str(e),
                metadata={"model_name": model_name}
            ) from e

        # Calculate metrics
        try:
            overall_metrics = metrics_calculator.calculate_metrics()
        except Exception as e:
            raise create_error_response(
                status_code=500,
                error_code="METRICS_CALCULATION_ERROR",
                message="Failed to calculate model metrics",
                technical_details=str(e),
                metadata={"model_name": model_name}
            ) from e

        # Calculate threshold metrics (if applicable)
        
        threshold_metrics = {}
        model_type = metadata.metadata.get('model_type', 'classification').lower()
        if model_type == "classification":
            try:
                threshold_metrics = metrics_calculator.calculate_threshold_metrics()
            except Exception as e:
                threshold_metrics = {
                    "status": "error",
                    "message": f"Failed to calculate threshold metrics: {e}",
                }

        # Combine results
        metrics = {
            "overall_metrics": overall_metrics,
            "threshold_metrics": threshold_metrics,
        }

        # Metrics must be JSON compliant. Our current test data contains Infinity values, which are not JSON compliant. We sanitize them to None.
        sanitized_metrics = sanitize_floats(metrics)

        return JSONResponse(
            content={
                "model_name": model_name,
                "metrics": sanitized_metrics,
                "docker_image_sha256": docker_image_sha256,
            }
        )

    except json.JSONDecodeError as e:
        raise create_error_response(
            status_code=400,
            error_code="INVALID_METADATA_JSON",
            message="Invalid model metadata JSON format",
            technical_details=str(e),
            metadata={"position": e.pos if hasattr(e, 'pos') else None}
        ) from e
    except HTTPException:
        # Re-raise HTTPException as-is (already structured)
        raise
    except Exception as e:
        raise create_error_response(
            status_code=500,
            error_code="MODEL_VALIDATION_FAILED",
            message="Model validation failed",
            technical_details=str(e)
        ) from e


def sanitize_floats(data: dict | list | float) -> dict | list | float | None:
    """
    Recursively sanitize floats in the data structure.
    Replace NaN and Infinity values with None.

    Parameters
    ----------
    data : dict | list | float
        The data structure to sanitize. It can be a dictionary, list, or float.

    Returns
    -------
    dict | list | float | None
        The sanitized data structure with NaN and Infinity values replaced by None.

    """

    if isinstance(data, dict):
        return {key: sanitize_floats(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [sanitize_floats(item) for item in data]
    elif isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return None  # Replace invalid float with `None`
    return data