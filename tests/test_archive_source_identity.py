"""Permanent archive source-identity contract tests."""

from shallweswim import config


def test_configured_citation_keys_match_archive_contract() -> None:
    """Require an explicit migration decision for any configured identity change."""
    actual = {
        location.code: {
            field: source.citation_key
            for field in (
                "live_temp_source",
                "historic_temp_source",
                "tide_source",
                "currents_source",
            )
            if (source := getattr(location, field)) is not None
        }
        for location in config.get_all_configs()
    }

    assert actual == {
        "aus": {
            "historic_temp_source": "nwis:temperature:08155500:00010",
            "live_temp_source": "nwis:temperature:08155500:00010",
        },
        "bos": {
            "historic_temp_source": "ndbc:temperature:44013",
            "live_temp_source": "ndbc:temperature:44013",
            "tide_source": "coops:tide:8443970",
        },
        "chi": {
            "historic_temp_source": "ndbc:temperature:45198",
            "live_temp_source": "ndbc:temperature:45198",
        },
        "cor": {
            "historic_temp_source": "irish-lights:temperature:992501100",
            "live_temp_source": "irish-lights:temperature:992501100",
            "tide_source": "marine-institute:tide:Kinsale",
        },
        "dov": {
            "historic_temp_source": "cspf:temperature:sandettie-data",
            "live_temp_source": "ndbc:temperature:62304",
            "tide_source": "local-harmonic:tide:data/tides/dov_harmonics.json",
        },
        "nyc": {
            "currents_source": "coops:currents:ACT3876,NYH1905",
            "historic_temp_source": "coops:temperature:8518750",
            "live_temp_source": "coops:temperature:8518750",
            "tide_source": "coops:tide:8517741",
        },
        "pbi": {
            "historic_temp_source": "coops:temperature:8722670",
            "live_temp_source": "coops:temperature:8722670",
            "tide_source": "coops:tide:8722670",
        },
        "san": {
            "historic_temp_source": "coops:temperature:9410230",
            "live_temp_source": "coops:temperature:9410230",
            "tide_source": "coops:tide:9410230",
        },
        "sdf": {"currents_source": "nwis:currents:03292494:72255"},
        "sea": {
            "historic_temp_source": "coops:temperature:9446484",
            "live_temp_source": "coops:temperature:9446484",
            "tide_source": "coops:tide:9447130",
        },
        "sfo": {
            "historic_temp_source": "ndbc:temperature:46237",
            "live_temp_source": "ndbc:temperature:46237",
            "tide_source": "coops:tide:9414305",
        },
        "tst": {
            "historic_temp_source": "ndbc:temperature:42001",
            "live_temp_source": "ndbc:temperature:42001",
        },
    }
