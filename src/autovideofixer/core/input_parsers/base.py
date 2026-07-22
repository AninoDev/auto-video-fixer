"""Auto Video Fixer - Input-list parser base class and registry.

Mirrors ``core/stages``' self-registration pattern (``BaseStage``/
``register_stage``/``get_stage``): each parser subclasses ``InputParser``,
sets a class-level ``name``, and calls ``register_parser(cls)`` at import
time. ``core/input_parsers/__init__.py`` imports every built-in parser
module so registration happens as a side effect of importing the package.

Parsers are pure and I/O-free (aside from parsing the text handed to them):
they never read files or resolve paths against the filesystem themselves --
that's the caller's job (see ``cli/cli.py``'s ``process`` command), using
``base_dir`` (the directory containing the list file) to resolve any
relative ``input_path``/``output_path`` the parser returns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class InputSpec:
    """One resolved entry from a parsed input-list file.

    ``input_path``/``output_path`` are returned exactly as parsed from the
    list file (i.e. possibly relative) -- the caller resolves them against
    ``base_dir``. ``recursive`` is ``None`` unless the list file itself
    specifies a per-entry override (e.g. the csv parser's `recursive`
    column); ``None`` means "use the run's --recursive flag" when this
    entry turns out to be a directory.
    """

    input_path: str
    output_path: str | None = None
    recursive: bool | None = None


class InputParser(ABC):
    """Abstract base class for input-list parsers.

    Subclasses set a class-level ``name`` (the string used to select this
    parser via ``--from-file-parser``/inline ``name:PATH``) and implement
    ``parse()``.
    """

    name: str = "base"

    @abstractmethod
    def parse(self, text: str, base_dir: str) -> list[InputSpec]:
        """Parse the full text of a list file into a list of InputSpecs.

        ``base_dir`` is the directory containing the list file, provided
        for parsers that need it to resolve relative paths themselves (the
        built-in parsers don't -- they return paths as-is and let the
        caller resolve them against ``base_dir``).
        """
        raise NotImplementedError


_PARSER_REGISTRY: dict[str, type[InputParser]] = {}


def register_parser(cls: type[InputParser]) -> type[InputParser]:
    """Register an InputParser subclass under its `name`. Usable as a decorator."""
    _PARSER_REGISTRY[cls.name] = cls
    return cls


def get_parser(name: str) -> InputParser:
    """Look up a registered parser by name and return an instance.

    Raises ValueError (with the list of known names) if `name` isn't registered.
    """
    cls = _PARSER_REGISTRY.get(name)
    if cls is None:
        known = ", ".join(sorted(_PARSER_REGISTRY)) or "(none registered)"
        raise ValueError(f"Unknown input-list parser {name!r}. Known parsers: {known}")
    return cls()


def list_parsers() -> dict[str, type[InputParser]]:
    """Return a copy of the parser registry: name -> InputParser subclass."""
    return dict(_PARSER_REGISTRY)
