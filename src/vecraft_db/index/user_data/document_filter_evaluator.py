from typing import Dict, Any, Callable, Tuple


class DocumentFilterEvaluator:
    """Evaluates filter conditions against document content."""

    def matches(self, document: str, condition: Dict[str, Any]) -> bool:
        """
        Check if document content matches the filter condition.
        Supports operators: $contains, $not_contains, $and, $or.
        """
        # Define for each operator: (handler, expected value type)
        ops: Dict[str, Tuple[Callable[[str, Any], bool], type]] = {
            "$contains": (
                lambda doc, val: val in doc,
                str
            ),
            "$not_contains": (
                lambda doc, val: val not in doc,
                str
            ),
            "$and": (
                lambda doc, sub_conditions: all(self.matches(doc, sc) for sc in sub_conditions),
                list
            ),
            "$or": (
                lambda doc, sub_conditions: any(self.matches(doc, sc) for sc in sub_conditions),
                list
            ),
        }

        for key, value in condition.items():
            handler_entry = ops.get(key)
            if not handler_entry:
                # Unknown operator → no match
                return False

            handler, expected_type = handler_entry
            if not isinstance(value, expected_type):
                # Wrong type for this operator → no match
                return False

            if not handler(document, value):
                return False

        return True