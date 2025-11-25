# Copyright 2025 Ilya Makarov
#
# Licensed under the Business Source License 1.1
# Change Date: 2028-11-05
# Change License: Apache License 2.0
# BBX String Adapter - String Operations

"""
BBX String Adapter

Provides string manipulation for BBX-only coding.

Methods:
    - split: Split string into array
    - join: Join array into string
    - replace: Replace substring
    - regex_match: Regex match
    - regex_replace: Regex replace
    - regex_find_all: Find all regex matches
    - upper/lower/title/capitalize: Case conversion
    - trim/strip: Remove whitespace
    - pad: Pad string
    - slice: Get substring
    - format: String formatting
    - template: Template substitution
    - encode/decode: Base64, URL encoding

Usage in .bbx:
    steps:
      parse_csv:
        use: string.split
        args:
          text: "a,b,c"
          separator: ","

      format_output:
        use: string.format
        args:
          template: "Hello, {name}!"
          values:
            name: Alice
"""

import base64
import json
import re
import urllib.parse
from typing import Any, Dict, List, Optional, Union

from blackbox.core.base_adapter import MCPAdapter


class StringAdapter(MCPAdapter):
    """
    String manipulation adapter for BBX workflows.

    Enables BBX-only coding without writing Python.
    """

    def __init__(self):
        super().__init__("string")

    async def execute(self, method: str, inputs: Dict[str, Any]) -> Any:
        """Execute string adapter method."""
        self.log_execution(method, inputs)

        method_map = {
            # Split/Join
            "split": self._split,
            "join": self._join,
            "lines": self._lines,

            # Replace
            "replace": self._replace,
            "replace_all": self._replace_all,

            # Regex
            "regex_match": self._regex_match,
            "regex_replace": self._regex_replace,
            "regex_find_all": self._regex_find_all,
            "regex_split": self._regex_split,

            # Case
            "upper": self._upper,
            "lower": self._lower,
            "title": self._title,
            "capitalize": self._capitalize,
            "swap_case": self._swap_case,

            # Trim
            "trim": self._trim,
            "trim_left": self._trim_left,
            "trim_right": self._trim_right,
            "strip": self._trim,  # Alias

            # Padding
            "pad_left": self._pad_left,
            "pad_right": self._pad_right,
            "center": self._center,

            # Substring
            "slice": self._slice,
            "substring": self._slice,  # Alias
            "char_at": self._char_at,

            # Search
            "contains": self._contains,
            "starts_with": self._starts_with,
            "ends_with": self._ends_with,
            "index_of": self._index_of,
            "count": self._count,

            # Format
            "format": self._format,
            "template": self._template,
            "concat": self._concat,
            "repeat": self._repeat,

            # Encoding
            "base64_encode": self._base64_encode,
            "base64_decode": self._base64_decode,
            "url_encode": self._url_encode,
            "url_decode": self._url_decode,
            "json_encode": self._json_encode,
            "json_decode": self._json_decode,

            # Conversion
            "to_int": self._to_int,
            "to_float": self._to_float,
            "to_bool": self._to_bool,

            # Utility
            "length": self._length,
            "reverse": self._reverse,
            "truncate": self._truncate,
            "slug": self._slug,
            "hash": self._hash,
        }

        handler = method_map.get(method)
        if not handler:
            raise ValueError(f"Unknown string method: {method}. Available: {list(method_map.keys())}")

        try:
            result = await handler(**inputs)
            self.log_success(method, result)
            return result
        except Exception as e:
            self.log_error(method, e)
            raise

    # === Split/Join ===

    async def _split(
        self,
        text: str,
        separator: str = " ",
        max_split: int = -1,
        remove_empty: bool = False,
    ) -> Dict[str, Any]:
        """Split string into array."""
        parts = text.split(separator, max_split) if max_split > 0 else text.split(separator)

        if remove_empty:
            parts = [p for p in parts if p]

        return {
            "status": "success",
            "result": parts,
            "count": len(parts),
        }

    async def _join(
        self,
        items: List[Any],
        separator: str = "",
    ) -> Dict[str, Any]:
        """Join array into string."""
        result = separator.join(str(item) for item in items)
        return {
            "status": "success",
            "result": result,
            "length": len(result),
        }

    async def _lines(
        self,
        text: str,
        keep_ends: bool = False,
    ) -> Dict[str, Any]:
        """Split text into lines."""
        if keep_ends:
            lines = text.splitlines(keepends=True)
        else:
            lines = text.splitlines()

        return {
            "status": "success",
            "result": lines,
            "count": len(lines),
        }

    # === Replace ===

    async def _replace(
        self,
        text: str,
        old: str,
        new: str,
        count: int = 1,
    ) -> Dict[str, Any]:
        """Replace first occurrence(s)."""
        result = text.replace(old, new, count)
        return {
            "status": "success",
            "result": result,
            "replaced": text != result,
        }

    async def _replace_all(
        self,
        text: str,
        old: str,
        new: str,
    ) -> Dict[str, Any]:
        """Replace all occurrences."""
        result = text.replace(old, new)
        count = text.count(old)
        return {
            "status": "success",
            "result": result,
            "count": count,
        }

    # === Regex ===

    async def _regex_match(
        self,
        text: str,
        pattern: str,
        flags: str = "",
    ) -> Dict[str, Any]:
        """Match regex pattern."""
        re_flags = self._parse_flags(flags)
        match = re.match(pattern, text, re_flags)

        if match:
            return {
                "status": "success",
                "matched": True,
                "match": match.group(0),
                "groups": match.groups(),
                "group_dict": match.groupdict(),
                "start": match.start(),
                "end": match.end(),
            }
        return {
            "status": "success",
            "matched": False,
        }

    async def _regex_replace(
        self,
        text: str,
        pattern: str,
        replacement: str,
        count: int = 0,
        flags: str = "",
    ) -> Dict[str, Any]:
        """Replace using regex."""
        re_flags = self._parse_flags(flags)
        result = re.sub(pattern, replacement, text, count=count, flags=re_flags)
        return {
            "status": "success",
            "result": result,
            "changed": result != text,
        }

    async def _regex_find_all(
        self,
        text: str,
        pattern: str,
        flags: str = "",
    ) -> Dict[str, Any]:
        """Find all regex matches."""
        re_flags = self._parse_flags(flags)
        matches = re.findall(pattern, text, re_flags)
        return {
            "status": "success",
            "matches": matches,
            "count": len(matches),
        }

    async def _regex_split(
        self,
        text: str,
        pattern: str,
        max_split: int = 0,
        flags: str = "",
    ) -> Dict[str, Any]:
        """Split using regex."""
        re_flags = self._parse_flags(flags)
        parts = re.split(pattern, text, maxsplit=max_split, flags=re_flags)
        return {
            "status": "success",
            "result": parts,
            "count": len(parts),
        }

    def _parse_flags(self, flags: str) -> int:
        """Parse regex flags string."""
        flag_map = {
            "i": re.IGNORECASE,
            "m": re.MULTILINE,
            "s": re.DOTALL,
            "x": re.VERBOSE,
        }
        result = 0
        for char in flags.lower():
            if char in flag_map:
                result |= flag_map[char]
        return result

    # === Case ===

    async def _upper(self, text: str) -> Dict[str, Any]:
        """Convert to uppercase."""
        return {"status": "success", "result": text.upper()}

    async def _lower(self, text: str) -> Dict[str, Any]:
        """Convert to lowercase."""
        return {"status": "success", "result": text.lower()}

    async def _title(self, text: str) -> Dict[str, Any]:
        """Convert to title case."""
        return {"status": "success", "result": text.title()}

    async def _capitalize(self, text: str) -> Dict[str, Any]:
        """Capitalize first letter."""
        return {"status": "success", "result": text.capitalize()}

    async def _swap_case(self, text: str) -> Dict[str, Any]:
        """Swap case."""
        return {"status": "success", "result": text.swapcase()}

    # === Trim ===

    async def _trim(self, text: str, chars: Optional[str] = None) -> Dict[str, Any]:
        """Strip whitespace from both ends."""
        result = text.strip(chars) if chars else text.strip()
        return {"status": "success", "result": result}

    async def _trim_left(self, text: str, chars: Optional[str] = None) -> Dict[str, Any]:
        """Strip from left."""
        result = text.lstrip(chars) if chars else text.lstrip()
        return {"status": "success", "result": result}

    async def _trim_right(self, text: str, chars: Optional[str] = None) -> Dict[str, Any]:
        """Strip from right."""
        result = text.rstrip(chars) if chars else text.rstrip()
        return {"status": "success", "result": result}

    # === Padding ===

    async def _pad_left(self, text: str, width: int, char: str = " ") -> Dict[str, Any]:
        """Pad from left."""
        return {"status": "success", "result": text.rjust(width, char[0])}

    async def _pad_right(self, text: str, width: int, char: str = " ") -> Dict[str, Any]:
        """Pad from right."""
        return {"status": "success", "result": text.ljust(width, char[0])}

    async def _center(self, text: str, width: int, char: str = " ") -> Dict[str, Any]:
        """Center string."""
        return {"status": "success", "result": text.center(width, char[0])}

    # === Substring ===

    async def _slice(
        self,
        text: str,
        start: int = 0,
        end: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Get substring."""
        result = text[start:end] if end is not None else text[start:]
        return {
            "status": "success",
            "result": result,
            "length": len(result),
        }

    async def _char_at(self, text: str, index: int) -> Dict[str, Any]:
        """Get character at index."""
        if 0 <= index < len(text):
            return {"status": "success", "result": text[index]}
        return {"status": "error", "error": f"Index out of range: {index}"}

    # === Search ===

    async def _contains(self, text: str, substring: str, case_sensitive: bool = True) -> Dict[str, Any]:
        """Check if contains substring."""
        if case_sensitive:
            result = substring in text
        else:
            result = substring.lower() in text.lower()
        return {"status": "success", "result": result}

    async def _starts_with(self, text: str, prefix: str) -> Dict[str, Any]:
        """Check if starts with prefix."""
        return {"status": "success", "result": text.startswith(prefix)}

    async def _ends_with(self, text: str, suffix: str) -> Dict[str, Any]:
        """Check if ends with suffix."""
        return {"status": "success", "result": text.endswith(suffix)}

    async def _index_of(self, text: str, substring: str, start: int = 0) -> Dict[str, Any]:
        """Find index of substring."""
        try:
            index = text.index(substring, start)
            return {"status": "success", "result": index, "found": True}
        except ValueError:
            return {"status": "success", "result": -1, "found": False}

    async def _count(self, text: str, substring: str) -> Dict[str, Any]:
        """Count occurrences of substring."""
        return {"status": "success", "result": text.count(substring)}

    # === Format ===

    async def _format(
        self,
        template: str,
        values: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Format string with values."""
        all_values = {**(values or {}), **kwargs}
        try:
            result = template.format(**all_values)
            return {"status": "success", "result": result}
        except KeyError as e:
            return {"status": "error", "error": f"Missing key: {e}"}

    async def _template(
        self,
        template: str,
        values: Optional[Dict[str, Any]] = None,
        delimiter: str = "${",
        **kwargs,
    ) -> Dict[str, Any]:
        """Template substitution with ${var} syntax."""
        all_values = {**(values or {}), **kwargs}
        result = template

        for key, value in all_values.items():
            placeholder = f"{delimiter}{key}}}"
            result = result.replace(placeholder, str(value))

        return {"status": "success", "result": result}

    async def _concat(self, *parts, separator: str = "") -> Dict[str, Any]:
        """Concatenate strings."""
        result = separator.join(str(p) for p in parts)
        return {"status": "success", "result": result}

    async def _repeat(self, text: str, count: int) -> Dict[str, Any]:
        """Repeat string."""
        return {"status": "success", "result": text * count}

    # === Encoding ===

    async def _base64_encode(self, text: str, encoding: str = "utf-8") -> Dict[str, Any]:
        """Encode to base64."""
        result = base64.b64encode(text.encode(encoding)).decode("ascii")
        return {"status": "success", "result": result}

    async def _base64_decode(self, text: str, encoding: str = "utf-8") -> Dict[str, Any]:
        """Decode from base64."""
        try:
            result = base64.b64decode(text).decode(encoding)
            return {"status": "success", "result": result}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _url_encode(self, text: str) -> Dict[str, Any]:
        """URL encode."""
        return {"status": "success", "result": urllib.parse.quote(text)}

    async def _url_decode(self, text: str) -> Dict[str, Any]:
        """URL decode."""
        return {"status": "success", "result": urllib.parse.unquote(text)}

    async def _json_encode(self, value: Any, indent: Optional[int] = None) -> Dict[str, Any]:
        """Encode to JSON string."""
        result = json.dumps(value, indent=indent, ensure_ascii=False, default=str)
        return {"status": "success", "result": result}

    async def _json_decode(self, text: str) -> Dict[str, Any]:
        """Decode JSON string."""
        try:
            result = json.loads(text)
            return {"status": "success", "result": result}
        except json.JSONDecodeError as e:
            return {"status": "error", "error": str(e)}

    # === Conversion ===

    async def _to_int(self, text: str, default: Optional[int] = None) -> Dict[str, Any]:
        """Convert to integer."""
        try:
            return {"status": "success", "result": int(text)}
        except ValueError:
            if default is not None:
                return {"status": "success", "result": default}
            return {"status": "error", "error": f"Cannot convert to int: {text}"}

    async def _to_float(self, text: str, default: Optional[float] = None) -> Dict[str, Any]:
        """Convert to float."""
        try:
            return {"status": "success", "result": float(text)}
        except ValueError:
            if default is not None:
                return {"status": "success", "result": default}
            return {"status": "error", "error": f"Cannot convert to float: {text}"}

    async def _to_bool(self, text: str) -> Dict[str, Any]:
        """Convert to boolean."""
        truthy = {"true", "1", "yes", "on", "y"}
        falsy = {"false", "0", "no", "off", "n", ""}
        lower = text.lower().strip()

        if lower in truthy:
            return {"status": "success", "result": True}
        if lower in falsy:
            return {"status": "success", "result": False}
        return {"status": "error", "error": f"Cannot convert to bool: {text}"}

    # === Utility ===

    async def _length(self, text: str) -> Dict[str, Any]:
        """Get string length."""
        return {"status": "success", "result": len(text)}

    async def _reverse(self, text: str) -> Dict[str, Any]:
        """Reverse string."""
        return {"status": "success", "result": text[::-1]}

    async def _truncate(
        self,
        text: str,
        max_length: int,
        suffix: str = "...",
    ) -> Dict[str, Any]:
        """Truncate string to max length."""
        if len(text) <= max_length:
            return {"status": "success", "result": text, "truncated": False}

        result = text[:max_length - len(suffix)] + suffix
        return {"status": "success", "result": result, "truncated": True}

    async def _slug(
        self,
        text: str,
        separator: str = "-",
        lowercase: bool = True,
    ) -> Dict[str, Any]:
        """Convert to URL slug."""
        # Remove non-alphanumeric
        result = re.sub(r"[^\w\s-]", "", text)
        # Replace whitespace with separator
        result = re.sub(r"[\s_]+", separator, result)
        # Remove duplicate separators
        result = re.sub(f"{separator}+", separator, result)
        # Strip separators from ends
        result = result.strip(separator)

        if lowercase:
            result = result.lower()

        return {"status": "success", "result": result}

    async def _hash(
        self,
        text: str,
        algorithm: str = "sha256",
    ) -> Dict[str, Any]:
        """Hash string."""
        import hashlib

        try:
            hash_func = hashlib.new(algorithm)
            hash_func.update(text.encode("utf-8"))
            return {
                "status": "success",
                "result": hash_func.hexdigest(),
                "algorithm": algorithm,
            }
        except ValueError:
            return {"status": "error", "error": f"Unknown algorithm: {algorithm}"}
