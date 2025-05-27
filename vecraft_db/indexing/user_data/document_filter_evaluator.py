"""
    Evaluates filter conditions against document content.

    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │                        DocumentFilterEvaluator Workflow                         │
    └─────────────────────────────────────────────────────────────────────────────────┘

    Input: document (str) + condition (Dict)
           │
           ▼
    ┌─────────────────────────────────────┐
    │  Initialize operator definitions    │ ◄─── Define handlers & expected types
    │  ops = {                            │      for $contains, $not_contains,
    │    "$contains": (handler, str),     │      $and, $or operators
    │    "$not_contains": (handler, str), │
    │    "$and": (handler, list),         │
    │    "$or": (handler, list)           │
    │  }                                  │
    └─────────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────────┐
    │   For each (key, value) pair        │ ◄─── Iterate through condition dict
    │   in condition.items():             │
    └─────────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────────┐     ┌─────────────────────────────────┐
    │   Look up operator in ops dict      │────►│  Unknown operator?              │
    │   handler_entry = ops.get(key)      │     │  if not handler_entry:          │
    └─────────────────────────────────────┘     │    return False                 │
           │                                    └─────────────────────────────────┘
           ▼                                                    │
    ┌─────────────────────────────────────┐                     │
    │   Extract handler and expected_type │                     │
    │   handler, expected_type =          │                     │
    │   handler_entry                     │                     │
    └─────────────────────────────────────┘                     │
           │                                                    │
           ▼                                                    │
    ┌─────────────────────────────────────┐     ┌─────────────────────────────────┐
    │   Validate value type               │────►│  Wrong type?                    │
    │   isinstance(value, expected_type)  │     │  if not isinstance(...):        │
    └─────────────────────────────────────┘     │    return False                 │
           │                                    └─────────────────────────────────┘
           ▼                                                    │
    ┌─────────────────────────────────────┐                     │
    │   Execute operator handler          │                     │
    │   handler(document, value)          │                     │
    └─────────────────────────────────────┘                     │
           │                                                    │
           ▼                                                    │
    ┌─────────────────────────────────────┐     ┌───────────────────────────────────────┐
    │   Handler returns boolean result    │────►│  Handler failed?                      │
    │                                     │     │  if not handler():                    │
    │   Operator-specific logic:          │     │    return False                       │
    │   • $contains: val in doc           │     └───────────────────────────────────────┘
    │   • $not_contains: val not in doc   │                          │
    │   • $and: all(recursive_calls)      │                          │
    │   • $or: any(recursive_calls)       │                          │
    └─────────────────────────────────────┘                          │
           │                                                         │
           ▼                                                         │
    ┌─────────────────────────────────────┐                          │
    │   Continue to next condition        │                          │
    │   (if handler succeeded)            │                          │
    └─────────────────────────────────────┘                          │
           │                                                         │
           ▼                                                         │
    ┌─────────────────────────────────────┐                          │
    │   All conditions processed?         │                          │
    │   End of condition.items() loop     │                          │
    └─────────────────────────────────────┘                          │
           │                                                         │
           ▼                                                         │
    ┌─────────────────────────────────────┐                          │
    │        SUCCESS PATH                 │                          │
    │     return True                     │ ◄────────────────────────┘
    │  (All conditions passed)            │
    └─────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │                           Operator Handler Details                              │
    ├─────────────────────────────────────────────────────────────────────────────────┤
    │                                                                                 │
    │  $contains:     lambda doc, val: val in doc                                     │
    │                 ├─ Input: document string, search value string                  │
    │                 └─ Returns: True if val found in doc                            │
    │                                                                                 │
    │  $not_contains: lambda doc, val: val not in doc                                 │
    │                 ├─ Input: document string, search value string                  │
    │                 └─ Returns: True if val NOT found in doc                        │
    │                                                                                 │
    │  $and:          lambda doc, sub_conditions: all(recursive_calls)                │
    │                 ├─ Input: document string, list of sub-conditions               │
    │                 ├─ Process: calls self.matches() for each sub-condition         │
    │                 └─ Returns: True only if ALL sub-conditions match               │
    │                                                                                 │
    │  $or:           lambda doc, sub_conditions: any(recursive_calls)                │
    │                 ├─ Input: document string, list of sub-conditions               │
    │                 ├─ Process: calls self.matches() for each sub-condition         │
    │                 └─ Returns: True if ANY sub-condition matches                   │
    │                                                                                 │
    └─────────────────────────────────────────────────────────────────────────────────┘
"""
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