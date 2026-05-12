"""Tonghuashun local board cache parser contract tests."""

import pytest

from qsss.data.manager import DataManager
from qsss.data.sources.ths_local_adapter import ThsLocalBoardAdapter


def test_ths_local_parser_reads_concept_members(tmp_path):
    """THS local concept cache should expose board member contract fields."""
    concept_path = tmp_path / "block_conception.ini"
    concept_path.write_text(
        "\n".join(
            [
                "[人工智能]",
                "300033=同花顺",
                "000977=浪潮信息",
                "",
                "[CPO]",
                "members=300308,688205",
                "",
                "[存储芯片]",
                "items=603986:兆易创新;300042:朗科科技",
            ]
        ),
        encoding="utf-8",
    )

    adapter = ThsLocalBoardAdapter(conception_path=concept_path)
    df = adapter.get_board_members("人工智能", board_type="concept")

    assert len(df) == 2
    assert list(df["symbol"]) == ["300033", "000977"]
    assert set(df["board_name"]) == {"人工智能"}
    assert set(df["board_type"]) == {"concept"}
    assert set(df["source"]) == {"ths_local_cache"}
    assert "version" in df.columns


def test_ths_local_parser_reads_industry_members(tmp_path):
    """THS local industry cache should parse industry boards independently."""
    industry_path = tmp_path / "block_industry.ini"
    industry_path.write_text("[智能电网]\nstocks=600406,300014\n", encoding="utf-8")

    adapter = ThsLocalBoardAdapter(industry_path=industry_path)
    df = adapter.get_board_members("智能电网", board_type="industry")

    assert list(df["symbol"]) == ["600406", "300014"]
    assert set(df["board_name"]) == {"智能电网"}
    assert set(df["board_type"]) == {"industry"}


def test_ths_local_parser_missing_path_is_explicit(tmp_path):
    """Missing THS cache files should fail explicitly."""
    adapter = ThsLocalBoardAdapter(conception_path=tmp_path / "missing.ini")

    with pytest.raises(FileNotFoundError, match="block_conception.ini"):
        adapter.get_board_members("人工智能", board_type="concept")


def test_data_manager_routes_registered_ths_board_members(tmp_path):
    """DataManager should expose board members through a registered THS adapter."""
    concept_path = tmp_path / "block_conception.ini"
    concept_path.write_text("[人工智能]\n300033=同花顺\n", encoding="utf-8")
    manager = DataManager()
    manager.register_adapter(
        "ths_local_cache", ThsLocalBoardAdapter(conception_path=concept_path)
    )

    df = manager.get_board_members("人工智能", board_type="concept")

    assert len(df) == 1
    assert df.iloc[0]["symbol"] == "300033"
