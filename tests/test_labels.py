import pytest

from src.fl.labels import GROUPS, to_group


@pytest.mark.parametrize(
    "label, group",
    [
        ("BenignTraffic", "Benign"),
        ("Benign_Final", "Benign"),
        ("DDoS-PSHACK_FLOOD", "DDoS"),
        ("DDoS-ICMP_Flood", "DDoS"),
        ("DoS-UDP_Flood", "DoS"),
        ("Mirai-greeth_flood", "Mirai"),
        ("Recon-PortScan", "Recon"),
        ("VulnerabilityScan", "Recon"),
        ("MITM-ArpSpoofing", "Spoofing"),
        ("DNS_Spoofing", "Spoofing"),
        ("SqlInjection", "Web-based"),
        ("DictionaryBruteForce", "BruteForce"),
    ],
)
def test_to_group(label, group):
    assert to_group(label) == group
    assert group in GROUPS


def test_unknown_label_raises():
    with pytest.raises(KeyError):
        to_group("NotALabel")
