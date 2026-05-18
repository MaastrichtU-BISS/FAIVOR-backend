from dataclasses import dataclass
import json
from typing import List, Dict, Any, Optional

@dataclass
class ModelInput:
    """
    Class to represent a model input feature.
    """
    input_label: str
    description: Optional[str] = None
    data_type: Optional[str] = None
    rdfs_label: Optional[str] = None

    def __post_init__(self):
        """
        Validate the input feature attributes.
        """
        if not isinstance(self.input_label, str):
            raise TypeError("input_label must be a string")
        if self.description is not None and not isinstance(self.description, str):
            raise TypeError("description must be a string")
        if self.data_type is not None and not isinstance(self.data_type, str):
            raise TypeError("data_type must be a string")
        if self.rdfs_label is not None and not isinstance(self.rdfs_label, str):
            raise TypeError("rdfs_label must be a string")

class ModelMetadata:
    """
    Class to store metadata information for a ML model, following FAIRmodels syntax.
    """

    def __init__(self, metadata_json: Dict[str, Any]):
        if not isinstance(metadata_json, dict):
            raise TypeError("Expected a dictionary for metadata_json")            
        self.metadata = metadata_json
        if "General Model Information" not in metadata_json:
            raise ValueError("Missing required 'General Model Information' section in metadata")
    
        self.inputs: List[ModelInput] = self._parse_inputs()
        self.output: str = self._parse_output()
        self.output_label: Optional[str] = self._extract_value(metadata_json.get("Outcome label"))

        general_info = self._normalize_section(self.metadata.get("General Model Information", {}))
        self.docker_image: Optional[str] = self._extract_value(general_info.get("FAIRmodels image name"))
        self.model_name: Optional[str] = self._extract_value(general_info.get("Title"))
        self.description: Optional[str] = self._extract_value(general_info.get("Editor Note"))
        self.author: Optional[str] = self._extract_value(general_info.get("Created by"))
        self.references: List[str] = self._extract_list_values(general_info.get("References to papers"))
        self.contact_email: Optional[str] = self._extract_value(general_info.get("Contact email"))

    @staticmethod
    def _normalize_section(section: Any) -> Dict[str, Any]:
        if isinstance(section, dict):
            return section
        if isinstance(section, list) and section and isinstance(section[0], dict):
            return section[0]
        return {}

    @staticmethod
    def _extract_value(value: Any) -> Optional[str]:
        if isinstance(value, dict):
            if "@value" in value:
                raw = value.get("@value")
                return str(raw) if raw is not None else None
            label = value.get("rdfs:label")
            if isinstance(label, dict):
                raw = label.get("@value")
                return str(raw) if raw is not None else None
            if isinstance(label, str):
                return label
            return None
        if isinstance(value, str):
            return value
        return None

    def _extract_list_values(self, values: Any) -> List[str]:
        if values is None:
            return []
        if not isinstance(values, list):
            values = [values]
        extracted: List[str] = []
        for item in values:
            parsed = self._extract_value(item)
            if parsed:
                extracted.append(parsed)
        return extracted

    def validate(self) -> bool:
        """
        Validate the metadata making sure it contains all required fields.
        
        Returns:
            bool: True if metadata is valid, False otherwise
        """
        required_fields = ["model_name", "inputs", "output"]
        return all(getattr(self, field) for field in required_fields)
    
    def _parse_inputs(self) -> List[ModelInput]:
        """
        Extract and normalize input features from the metadata.
        
        Returns:
            List of ModelInput objects containing normalized input feature information.
        """
        
        inputs: List[ModelInput] = []
        # prefer "Input data", fallback to "Input data1"
        inputs_data = self.metadata.get("Input data") or self.metadata.get("Input data1") or []
        if not isinstance(inputs_data, list):
            inputs_data = [inputs_data]
        for input_feature in inputs_data:
            if not isinstance(input_feature, dict):
                continue
            feature = ModelInput(
                input_label = self._extract_value(input_feature.get("Input label")) or "",
                description = self._extract_value(input_feature.get("Description")),
                data_type = self._extract_value(input_feature.get("Type of input")),
                rdfs_label = self._extract_value(input_feature.get("Input feature"))
            )
            inputs.append(feature)
        return inputs

    def _parse_output(self) -> str:
        """
        Extract the model's output label from the metadata.
        
        Returns:
            str: The output label value from the 'Outcome label' field
        """
                
        return self._extract_value(self.metadata.get("Outcome label")) or ""

    def __repr__(self) -> str:
        """
        Create a string representation of the class object.
        
        Returns a JSON string containing the key attributes of the
        model metadata
        
        Returns:
            str: JSON string representation of the model metadata
        """
                
        return json.dumps({
            "model_name": self.model_name,
            "description": self.description,
            "docker_image": self.docker_image,
            "inputs": self.inputs,
            "output": self.output,
            "output_label": self.output_label,
            "author": self.author,
            "references": self.references,
            "contact_email": self.contact_email
        }, indent=2)

