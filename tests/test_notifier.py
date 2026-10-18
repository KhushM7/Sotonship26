from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from fallsense_server.incidents import Push
from fallsense_server.notifier import Notifier, Tokens
from fallsense_server.settings import Carer, Settings


def test_notification_links_survive_spaces_in_names(tmp_path: Path) -> None:
    settings = Settings(
        raw={
            "ntfy": {"server": "https://ntfy.sh"},
            "public_url": "http://10.0.0.5:8000",
        },
        carers=[Carer("Amara Okafor", 1, topic="t")],
        rooms=[],
    )
    tokens = Tokens(tmp_path / "secret.key")
    body = Notifier(settings, tokens).payload(
        Push(["Amara Okafor"], 5, "FALL – Room 4", "Fall detected", "INC-0001"),
        "Amara Okafor",
        "t",
    )
    urls = [body["click"], *(a["url"] for a in body["actions"])]
    for url in urls:
        assert " " not in url
        q = parse_qs(urlsplit(url).query)
        assert q["carer"] == ["Amara Okafor"]
        assert tokens.check("Amara Okafor", q["token"][0])
    going = body["actions"][0]
    assert going["action"] == "view"
    assert going["url"].startswith("http://10.0.0.5:8000/phone/i/INC-0001?")
    assert parse_qs(urlsplit(going["url"]).query)["ack"] == ["1"]
