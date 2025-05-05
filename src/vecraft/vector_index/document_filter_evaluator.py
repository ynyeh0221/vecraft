from typing import Dict, Any


class DocumentFilterEvaluator:
    """Evaluates filter conditions against document content."""

    def matches(self, document: str, condition: Dict[str, Any]) -> bool:
        """Check if document content matches the filter condition."""
        for key, value in condition.items():
            if key == "$contains" and isinstance(value, str):
                if value not in document:
                    return False
            elif key == "$not_contains" and isinstance(value, str):
                if value in document:
                    return False
            elif key == "$and" and isinstance(value, list):
                if not all(self.matches(document, subcond) for subcond in value):
                    return False
            elif key == "$or" and isinstance(value, list):
                if not any(self.matches(document, subcond) for subcond in value):
                    return False
            else:
                # Unsupported operator
                return False
        return True