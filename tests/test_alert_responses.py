"""The alert responses have one wording, in three places that must agree.

`infra/monitoring/alert_responses.json` is the source: Terraform builds each
alert policy's notification subject and body from it. The table in
infra/MONITORING.md "Responding to alerts" repeats it for a reader, and every
alert policy in the Terraform module must have an entry.
"""

import json
import re
from pathlib import Path

MONITORING = Path(__file__).parent.parent / "infra" / "monitoring"
MONITORING_DOC = MONITORING.parent / "MONITORING.md"
RESPONSE_FIELDS = {"policy", "urgency", "title", "action"}


def _responses() -> dict[str, dict[str, str]]:
    responses: dict[str, dict[str, str]] = json.loads(
        (MONITORING / "alert_responses.json").read_text()
    )
    return responses


def _terraform_policies() -> dict[str, str]:
    """Return each alert policy's display name by its Terraform resource name."""
    policies: dict[str, str] = {}
    for path in MONITORING.glob("*.tf"):
        policies.update(
            re.findall(
                r'resource "google_monitoring_alert_policy" "(\w+)" \{\s+'
                r'display_name\s+= "\[Terraform\] ([^"]+)"',
                path.read_text(),
            )
        )
    return policies


def _documented_rows() -> dict[str, tuple[str, str]]:
    """Return the document table's urgency and action by policy name."""
    section = MONITORING_DOC.read_text().split(
        "### What each policy asks of the operator"
    )[1]
    table = section.split("| --- | --- | --- |\n")[1].split("\n\n")[0]
    rows: dict[str, tuple[str, str]] = {}
    for line in table.splitlines():
        policy, urgency, action = (cell.strip() for cell in line.strip("|").split("|"))
        rows[policy] = (urgency, action)
    return rows


def test_every_response_is_complete() -> None:
    for key, response in _responses().items():
        assert set(response) == RESPONSE_FIELDS, key
        assert all(response.values()), key


def test_every_terraform_policy_has_its_response() -> None:
    assert _terraform_policies() == {
        key: response["policy"] for key, response in _responses().items()
    }


def test_every_terraform_policy_notifies_with_its_response() -> None:
    terraform = "".join(path.read_text() for path in MONITORING.glob("*.tf"))
    for key in _responses():
        assert f"local.alert_subject.{key}\n" in terraform, key
        assert f"local.alert_guidance.{key},\n" in terraform, key


def test_the_document_table_repeats_the_responses_exactly() -> None:
    assert _documented_rows() == {
        response["policy"]: (response["urgency"], response["action"])
        for response in _responses().values()
    }
