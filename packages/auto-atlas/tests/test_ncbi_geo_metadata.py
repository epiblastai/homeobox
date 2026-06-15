from auto_atlas.ncbi import (
    GeoSampleMetadata,
    GeoSeriesMetadata,
    fetch_geo_sample,
    geo_metadata_to_dict,
)


def test_geo_metadata_to_dict_for_series_preserves_legacy_shape() -> None:
    metadata = GeoSeriesMetadata(
        accession="GSE123",
        title="Series title",
        summary="Series summary",
        organism="Homo sapiens",
        n_samples=2,
        platform_ids=["GPL1", "GPL2"],
        ftp_link="ftp://example",
        pmids=["12345"],
        doi="10.1000/example",
        bioproject="PRJNA1",
        sra_accession="SRP1",
        samples=[{"accession": "GSM1", "title": "Sample"}],
    )

    assert geo_metadata_to_dict(metadata) == {
        "accession": "GSE123",
        "title": "Series title",
        "summary": "Series summary",
        "organism": "Homo sapiens",
        "n_samples": 2,
        "platform": "GPL1;GPL2",
        "ftp_link": "ftp://example",
        "pmids": ["12345"],
        "doi": "10.1000/example",
    }


def test_geo_metadata_to_dict_for_sample_preserves_legacy_shape() -> None:
    metadata = GeoSampleMetadata(
        accession="GSM123",
        title="Sample title",
        source="cells",
        organism="Mus musculus",
        characteristics={"cell type": "T cell", "treatment": "control"},
        molecule="total RNA",
        platform="GPL1",
        description="Sample description",
        treatment_protocol="Treatment",
        growth_protocol="Growth",
        extract_protocol="Extract",
        data_processing="Processing",
        series_ids=["GSE123"],
        biosample_accession="SAMN1",
    )

    assert geo_metadata_to_dict(metadata) == {
        "accession": "GSM123",
        "title": "Sample title",
        "source": "cells",
        "organism": "Mus musculus",
        "characteristics": {"cell type": "T cell", "treatment": "control"},
        "molecule": "total RNA",
        "platform": "GPL1",
        "description": "Sample description",
        "treatment_protocol": "Treatment",
        "growth_protocol": "Growth",
        "extract_protocol": "Extract",
        "data_processing": "Processing",
        "gse": ["GSE123"],
    }


def test_fetch_geo_sample_preserves_characteristic_key_labels(monkeypatch) -> None:
    soft_text = "\n".join(
        [
            "!Sample_title = Sample title",
            "!Sample_characteristics_ch1 = Cell type: T cell",
            "!Sample_characteristics_ch1 = perturbation",
        ]
    )

    monkeypatch.setattr("auto_atlas.ncbi._geo_soft_get", lambda accession: soft_text)

    metadata = fetch_geo_sample("GSM123")

    assert metadata.characteristics == {
        "Cell type": "T cell",
        "perturbation": "perturbation",
    }
