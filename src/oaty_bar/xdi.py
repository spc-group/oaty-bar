"""This module holds tools for parsing text formatted in the XAS data
interchange format.

For details of the specification, see:
https://github.com/XraySpectroscopy/XAS-Data-Interchange/

Forward parsing takes place in two steps:

1. Convert text into tokens (``tokenize()``)
2. Parse the tokens into a dataset (``parse()``)

These steps are combine in the ``load()`` function.

For going in the other direction, use ``dump()``.

"""

__all__ = ["load", "dump"]

import ast
import re
from collections.abc import Generator, Iterable
from dataclasses import dataclass
from enum import Enum
from itertools import chain
from pathlib import Path
from typing import Literal, TypedDict, cast

import xarray as xr
from xarray.backends import BackendEntrypoint


class XDIMalformed(Exception):
    pass


class Role(int, Enum):
    VERSION = 0
    HEADER_NAME = 1
    HEADER_VALUE = 2
    USER_COMMENT = 3
    COLUMN_LABEL = 4
    DATUM = 5


class Attrs(TypedDict):
    xdi_version: Literal["", "1.0"]
    versions: dict[str, str]
    header: dict[str, str]
    user_comment: str


@dataclass()
class Token:
    value: str
    role: Role


number_pattern = re.compile("[-_0-9.e]+")


def as_number(value: str) -> float | int:
    # Make sure only valid number characters are present
    if number_pattern.match(value):
        return ast.literal_eval(value)
    else:
        return float("nan")


field_end_pattern = re.compile(r"#\s*///+\s*")
header_end_pattern = re.compile(r"#\s*---+\s*")
version_pattern = re.compile(r"#\s*(XDI/[^ \t]+)((?:[ \t]+[^ \t/]+/[^ \t/]+)*)\s*")
header_pattern = re.compile(r"#\s*([^:]+):(.+)")
user_comment_pattern = re.compile(r"#\s*(.*)")
space_separated_pattern = re.compile("[^ \t]+")  # Individual labels, not the whole line


def tokenize(text: str, strict: bool) -> Generator[Token]:
    """Lexxer that turns an XDI formatted text into Line tokens."""
    lax = not strict
    current_section = "version"
    for line in text.split("\n"):
        # Check for field- and header-end lines
        if match := field_end_pattern.match(line):
            current_section = "user_comments"
            continue
        elif match := header_end_pattern.match(line):
            current_section = "data"
            continue
        # Check for a version line
        elif current_section == "version" and (match := version_pattern.match(line)):
            current_section = "header"
            grp1, grp2 = match.groups()
            versions = [grp1, *grp2.strip().split(" ")]
            versions = [version for version in versions if version != ""]
            yield from (Token(ver, Role.VERSION) for ver in versions)
            continue
        # Check for header fields
        elif (current_section == "header" or lax) and (
            match := header_pattern.match(line)
        ):
            name, value = match.groups()
            yield Token(name.strip(), Role.HEADER_NAME)
            yield Token(value.strip(), Role.HEADER_VALUE)
            continue
        # Check for user comments
        elif current_section == "user_comments" and (
            match := user_comment_pattern.match(line)
        ):
            yield Token(match.group(1), Role.USER_COMMENT)
        # Column labels
        elif (current_section == "data") and line.startswith("#"):
            for match in space_separated_pattern.finditer(line.lstrip("#")):
                yield Token(match.group(), Role.COLUMN_LABEL)
        elif current_section == "data":
            for match in space_separated_pattern.finditer(line):
                yield Token(match.group(), Role.DATUM)
        else:
            msg = f"Cannot parse line '{line}'. Consider using *strict=False*."
            raise XDIMalformed(msg)


def parse(tokens: Iterable[Token], strict: bool) -> xr.Dataset:
    """Convert tokens from ``tokenize()`` into a dataset."""
    tokens_ = iter(tokens)
    labels = []
    data = []
    # Check that the XDI version is first
    token = next(tokens_)
    if token.role == Role.VERSION and token.value.startswith("XDI/"):
        xdi_version = token.value.split("/")[1]
    elif strict:
        raise XDIMalformed(f"Invalid version token {token}.")
    else:
        # It's not actually a version line, so re-pack it so it can be
        # parsed properly
        xdi_version = ""
        tokens_ = chain([token], tokens_)
    if xdi_version != "" and xdi_version != "1.0":
        raise XDIMalformed(f"Unknown XDI version: {xdi_version}")
    else:
        xdi_version = cast(Literal["", "1.0"], xdi_version)
    # Process the rest of the tokens
    user_comments = []
    headers = {}
    versions = {}
    while True:
        try:
            token = next(tokens_)
        except StopIteration:
            break
        if token.role == Role.VERSION:
            package, version = token.value.split("/")
            versions[package] = version
        elif token.role == Role.HEADER_NAME:
            # Header name, so next token should be the value
            headers[token.value] = next(tokens_).value
        elif token.role == Role.USER_COMMENT:
            # Combine user comments
            # comment = "\n".join([user_comment, token.value]).strip("\n")
            user_comments.append(token.value)
        elif token.role == Role.COLUMN_LABEL:
            labels.append(token.value)
        elif token.role == Role.DATUM:
            data.append(as_number(token.value))
        else:
            raise XDIMalformed(f"Unknown token role: {token}")
    # Align the data with their column labels
    attrs: Attrs = dict(
        xdi_version=xdi_version,
        versions=versions,
        header=headers,
        user_comment="\n".join(user_comments),
    )
    if len(labels) == 0:
        coords = {}
        data_vars = {}
    else:
        coords = {labels[0]: data[:: len(labels)]}
        dv = {
            label: data[idx + 1 :: len(labels)] for idx, label in enumerate(labels[1:])
        }
        data_vars = {label: (coords.keys(), data) for label, data in dv.items()}
    ds = xr.Dataset(
        coords=coords,
        data_vars=data_vars,
        attrs=attrs,
    )
    # Add unit metadata from column headers
    vals = [
        val.split(" ") for key, val in headers.items() if key.split(".")[0] == "Column"
    ]
    unit_vals = [val for val in vals if len(val) == 2]
    units = {col: unit for col, unit in unit_vals}
    for col, unit in units.items():
        ds[col].attrs["units"] = unit
    return ds


def load(xdi_text: str, strict: bool = True) -> xr.Dataset:
    """Convert an XDI formatted string to an Xarray.

    Non-data parts of the XDI file are included in the dataset's
    ``attrs`` attribute. Namely:

    "xdi_version"
      Holds the XDI version specifier, e.g. ``"1.0"``
    "versions"
      Mapping of package names to version specifiers,
      e.g. ``{"GSE":"1.0"}``.
    "header"
      A mapping of header names to values,
      e.g. ``{"Column.1": "energy eV"}``
    "user_comment"
      Free form text that will go in the user comment section of the
      XDI output.

    The array labels are taken from the column labels section of the
    XDI text. Headers like ``"Column.1"`` are left as is in the
    *dataset*; it is left up to the client to interpret these headers properly.

    Parameters
    ==========
    xdi_text
      A string representing the input dataset in XDI format, suitable
      for passing the an open file objects ``.write()`` method.
    strict
      If true, files must meet the XDI specification or else an
      exception will be raised.

    Returns
    =======
    dataset
      An Xarray dataset with the data to save.

    """
    tokens = tokenize(xdi_text, strict=strict)
    dataset = parse(tokens, strict=strict)
    return dataset


class XDIBackendEntrypoint(BackendEntrypoint):
    description = "Use .xdi files in Xarray"
    url = "https://github.com/spc-group/hollowfoot?tab=readme-ov-file#xas-data-interchange-format-xdi"
    open_dataset_parameters = ["filename_or_obj", "drop_variables"]

    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables=None,
        # other backend specific keyword arguments
        # `chunks` and `cache` DO NOT go here, they are handled by xarray
    ):
        with open(filename_or_obj) as fp:
            dataset = load(fp.read())
        return dataset

    def guess_can_open(self, filename_or_obj):
        ext = Path(filename_or_obj).suffix
        return ext in {".xdi"}


def dump(dataset: xr.Dataset, strict: bool = True) -> str:
    """Convert an Xarray to an XDI formatted string.

    Depends on the xarray having the correct attrs to produce an XDI
    file. Namely:

    "xdi_version"
      Holds the XDI version specifier, e.g. ``"1.0"``
    "versions"
      Mapping of package names to version specifiers,
      e.g. ``{"GSE":"1.0"}``.
    "header"
      A mapping of header names to values,
      e.g. ``{"Column.1": "energy eV"}``
    "user_comment"
      Free form text that will go in the user comment section of the
      XDI output.

    The column labels section and `"ColumnN.name"` headers are taken
    from the names of the arrays in the dataset. If an array has the
    `"units"` attr, that will be used in the `"ColumnN.name"` header
    value.

    Parameters
    ==========
    dataset
      An Xarray dataset with the data to save.

    Returns
    =======
    xdi
      A string representing the input dataset in XDI format, suitable
      for passing the an open file objects ``.write()`` method.
    strict
      If true, a valid XDI file will always be generated, or an
      exception will be raised if a valid XDI file cannot be
      produced. If false, a file will always be created even if it
      isn't a valid XDI file.

    """
    lines: list[str] = []
    # Version specifiers
    other_versions = [
        f"{name}/{version}"
        for name, version in dataset.attrs.get("versions", {}).items()
    ]
    if dataset.attrs.get("xdi_version", "") or strict:
        xdi_version = f"# XDI/{dataset.attrs['xdi_version']}"
        lines.extend([" ".join([*(xdi_version, *other_versions)])])
    # Header metadata
    headers = dataset.attrs.get("header", {})
    ds_items = chain(dataset.coords.items(), dataset.data_vars.items())
    for idx, (name, arr) in enumerate(ds_items):
        col_name = f"{name} {arr.attrs.get('units', '')}".strip()
        headers.setdefault(f"Column.{idx+1}", col_name)
    lines.extend([f"# {key}: {val}" for key, val in headers.items()])
    # User comments section
    user_comment = dataset.attrs.get("user_comment", "").strip()
    if bool(user_comment):
        lines.append("# /////")
        lines.extend([f"# {line}" for line in user_comment.split("\n")])
    # Column headers
    lines.append("# -----")
    labels = [*dataset.coords.keys(), *dataset.data_vars.keys()]
    lines.append(f"# {'\t'.join(labels)}")
    # Data in tab-separated format
    df = dataset.to_dataframe().reset_index()
    data_rows = df.to_csv(sep="\t", header=False, index=False).split("\n")
    data_rows = [f"  {row}" for row in data_rows]
    lines.extend(data_rows)
    # Clean up the file ending (make sure there's a newline)
    text = "\n".join(lines).rstrip() + "\n"
    return text
