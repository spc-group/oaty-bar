from oaty_bar._export_run import main


def test_export_hdf(tmp_path, xafs_run, mocker, prefect_server):
    from_profile = mocker.MagicMock(
        return_value={
            "123-45-67890": xafs_run,
        }
    )
    mocker.patch("oaty_bar._export_run.from_profile", new=from_profile)

    # Wrap in a flow to make prefect happy
    main(
        [
            "123-45-67890",
            str(tmp_path),
            "--raw-profile",
            "raw_catalog",
            # "--results-profile",
            # "proc_catalog",
        ]
    )
    # Check that the file was created
    target_file = tmp_path / "202210060914-NMC-811-Pristine-rel_scan-7d1daf1d.h5"
    assert target_file.exists()
