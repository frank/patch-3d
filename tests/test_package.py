import patch_pipeline


def test_package_exposes_version() -> None:
    assert patch_pipeline.__version__


def test_package_exports_both_pipelines() -> None:
    assert set(patch_pipeline.__all__) == {"PatchPipeline", "RoiPipeline", "__version__"}
