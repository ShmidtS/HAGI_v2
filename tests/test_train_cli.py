from pathlib import Path

import scripts.train as train_cli


def test_bundled_default_config_resolves_its_data_paths() -> None:
    args = train_cli.build_parser().parse_args([])

    assert args.config == train_cli.DEFAULT_CONFIG
    assert args.config.is_file()

    overrides = train_cli.default_config_path_overrides(args.config)
    assert Path(overrides["model.head.unigram_path"]).is_file()
    assert Path(overrides["train.data.data_dir"]).is_dir()
    assert Path(overrides["train.checkpoint_dir"]).is_absolute()


def test_external_config_paths_are_not_rewritten(tmp_path: Path) -> None:
    external = tmp_path / "external.yaml"
    external.write_text("{}", encoding="utf-8")

    assert train_cli.default_config_path_overrides(external) == {}
