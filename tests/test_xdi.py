from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from oaty_bar.xdi import (
    Role,
    Token,
    XDIBackendEntrypoint,
    dump,
    load,
    parse,
    tokenize,
    version_pattern,
)

xdi_path = Path(__file__).parent / "example.xdi"
tsv_path = Path(__file__).parent / "example.tsv"


def test_tokenizer_strict():
    with open(xdi_path, mode="r") as fp:
        tokens = list(tokenize(fp.read(), strict=True))

    assert tokens[0].role == Role.VERSION
    assert tokens[0].value == "XDI/1.0"
    assert tokens[1].role == Role.VERSION
    assert tokens[1].value == "GSE/1.0"
    assert tokens[2].role == Role.HEADER_NAME
    assert tokens[2].value == "Column.1"
    assert tokens[46].role == Role.USER_COMMENT
    assert tokens[46].value == "Cu foil Room Temperature"
    expected_columns = ["energy", "i0", "itrans", "mutrans"]
    column_labels = tokens[48:52]
    assert [col.role for col in column_labels] == [Role.COLUMN_LABEL] * len(
        expected_columns
    )
    assert [col.value for col in column_labels] == expected_columns
    assert tokens[52].role == Role.DATUM
    assert tokens[52].value == "8779.0"


def test_tokenizer_lax():
    with open(tsv_path, mode="r") as fp:
        tokens = list(tokenize(fp.read(), strict=False))
    assert tokens[0].role == Role.HEADER_NAME
    assert tokens[0].value == "Column.1"
    assert tokens[10].role == Role.COLUMN_LABEL
    assert tokens[10].value == "energy"
    expected_columns = ["energy", "i0", "itrans", "mutrans"]
    column_labels = tokens[10:14]
    assert [col.role for col in column_labels] == [Role.COLUMN_LABEL] * len(
        expected_columns
    )
    assert [col.value for col in column_labels] == expected_columns
    assert tokens[14].role == Role.DATUM
    assert tokens[14].value == "8779.0"


version_lines = [
    # (text, number of versions)
    ("# Sample name: Blah blah", 0),
    ("# XDI/1.0", 1),
    ("# XDI/1.0 python/3.14.0", 2),
    ("# XDI/1.0 python/3.14.0 hollowfoot/2025.2rc23", 3),
]


@pytest.mark.parametrize("text,count", version_lines)
def test_version_pattern(text, count):
    match = version_pattern.match(text)
    if count == 0:
        pass
    else:
        assert match
        grp1, grp2 = match.groups()
        versions = [grp1, *grp2.strip().split(" ")]
        versions = [version for version in versions if version != ""]
        assert len(versions) == count


@pytest.mark.parametrize("text,count", version_lines)
def test_version_pattern(text, count):
    match = version_pattern.match(text)
    if count == 0:
        pass
    else:
        assert match
        grp1, grp2 = match.groups()
        versions = [grp1, *grp2.strip().split(" ")]
        versions = [version for version in versions if version != ""]
        assert len(versions) == count


def test_parse_version():
    dataset = parse(
        [Token("XDI/1.0", Role.VERSION), Token("GSE/1.3", Role.VERSION)], strict=True
    )
    assert dataset.attrs["xdi_version"] == "1.0"
    assert dataset.attrs["versions"]["GSE"] == "1.3"


def test_parse_header():
    dataset = parse(
        [
            Token("XDI/1.0", Role.VERSION),
            Token("Mono.d_spacing", Role.HEADER_NAME),
            Token("3.687", Role.HEADER_VALUE),
        ],
        strict=True,
    )
    assert dataset.attrs["header"]["Mono.d_spacing"] == "3.687"


def test_parse_user_comment():
    dataset = parse(
        [
            Token("XDI/1.0", Role.VERSION),
            Token("spam", Role.USER_COMMENT),
            Token("and eggs", Role.USER_COMMENT),
        ],
        strict=True,
    )
    assert dataset.attrs["user_comment"] == "spam\nand eggs"


def test_parse_column_labels():
    tokens = [
        Token("XDI/1.0", Role.VERSION),
        Token("mono-energy", Role.COLUMN_LABEL),
        Token("It-net_count", Role.COLUMN_LABEL),
    ]
    dataset = parse(tokens, strict=True)
    assert list(dataset.coords.keys()) == ["mono-energy"]
    assert list(dataset.keys()) == ["It-net_count"]


def test_parse_data():
    tokens = [
        Token("XDI/1.0", Role.VERSION),
        Token("mono-energy", Role.COLUMN_LABEL),
        Token("It-net_count", Role.COLUMN_LABEL),
        Token("8333.0", Role.DATUM),
        Token("54893992", Role.DATUM),
    ]
    dataset = parse(tokens, strict=True)
    np.testing.assert_equal(dataset["mono-energy"].values, [8333.0])
    np.testing.assert_equal(dataset["It-net_count"].values, [54893992])


def test_load_strict():
    with open(xdi_path, mode="r") as fp:
        dataset = load(fp.read(), strict=True)
    assert isinstance(dataset, xr.Dataset)
    assert dataset.attrs["header"]["Facility.xray_source"] == "APS Undulator A"


def test_load_lax():
    with open(tsv_path, mode="r") as fp:
        dataset = load(fp.read(), strict=False)
    assert isinstance(dataset, xr.Dataset)
    assert dataset.attrs["header"]["Column.1"] == "energy eV"


def test_load_nonstrict():
    with open(xdi_path, mode="r") as fp:
        dataset = load(fp.read(), strict=False)


def test_xarray_plugin():
    assert XDIBackendEntrypoint().guess_can_open(xdi_path)
    dataset = xr.open_dataset(xdi_path)
    assert isinstance(dataset, xr.Dataset)
    assert dataset.attrs["header"]["Facility.xray_source"] == "APS Undulator A"


strict_dataset = xr.Dataset(
    coords={
        "energy": np.linspace(8779, 8889, num=12),
    },
    data_vars={
        "i0": (
            "energy",
            [
                149013.7,
                144864.7,
                132978.7,
                125444.7,
                121324.7,
                119447.7,
                119100.7,
                117707.7,
                117754.7,
                117428.7,
                117383.7,
                117185.7,
            ],
        ),
        "itrans": (
            "energy",
            [
                550643.089065,
                531876.119084,
                489591.10592,
                463051.104096,
                449969.103983,
                444386.117562,
                440176.091039,
                440448.106567,
                442302.10637,
                441944.116528,
                442810.120466,
                443658.11566,
            ],
        ),
        "mutrans": (
            "energy",
            [
                -1.3070486,
                -1.3006104,
                -1.3033816,
                -1.3059724,
                -1.3107085,
                -1.3138152,
                -1.3072055,
                -1.3195882,
                -1.3233895,
                -1.3253521,
                -1.327693,
                -1.3312944,
            ],
        ),
    },
    attrs={
        "xdi_version": "1.0",
        "versions": {
            "SPC": "1.3",
        },
        "header": {
            "Column.1": "energy eV",
        },
        "user_comment": "Hello\nspam and eggs\n",
    },
)

minimal_dataset = xr.Dataset(
    data_vars={
        "i0": [
            149013.7,
            144864.7,
            132978.7,
            125444.7,
            121324.7,
            119447.7,
            119100.7,
            117707.7,
            117754.7,
            117428.7,
            117383.7,
            117185.7,
        ],
    },
)


def test_dump_strict():
    xdi = dump(strict_dataset, strict=True)
    lines = xdi.split("\n")
    assert lines[0] == "# XDI/1.0 SPC/1.3"
    assert lines[1] == "# Column.1: energy eV"
    assert lines[2] == "# /////"
    assert lines[3] == "# Hello"
    assert lines[4] == "# spam and eggs"
    assert lines[5] == "# -----"
    assert lines[6] == "# energy\ti0\titrans\tmutrans"
    assert lines[7] == "  8779.0\t149013.7\t550643.089065\t-1.3070486"
    # Make sure there a trailing newline
    assert lines[-2] == "  8889.0\t117185.7\t443658.11566\t-1.3312944"
    assert lines[-1] == ""


def test_dump_lax():
    xdi = dump(minimal_dataset, strict=False)
    lines = xdi.split("\n")
    assert lines[0] == "# -----"
    assert lines[1] == "# i0"
    assert lines[2] == "  149013.7"


def test_dump_sparse_dataset():
    """What is the minimal set of parameters that can still create valid XDI."""
    sparse_dataset = xr.Dataset(
        data_vars={"values": [5]},
        attrs={"xdi_version": "1.0"},
    )
    xdi = dump(sparse_dataset)
    assert xdi == "# XDI/1.0\n# -----\n# values\n  5\n"
