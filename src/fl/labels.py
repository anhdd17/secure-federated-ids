"""CICIoT2023 label mapping: 34 fine-grained labels -> 8 groups (Neto et al., 2023)."""

GROUPS = ["Benign", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web-based", "BruteForce"]

# Labels whose group cannot be read from a prefix.
_EXPLICIT = {
    "BenignTraffic": "Benign",
    "Benign_Final": "Benign",  # folder name in the UNB CSV.zip release
    "MITM-ArpSpoofing": "Spoofing",
    "DNS_Spoofing": "Spoofing",
    "VulnerabilityScan": "Recon",
    "DictionaryBruteForce": "BruteForce",
    "SqlInjection": "Web-based",
    "XSS": "Web-based",
    "CommandInjection": "Web-based",
    "Backdoor_Malware": "Web-based",
    "Uploading_Attack": "Web-based",
    "BrowserHijacking": "Web-based",
}

_PREFIXES = {"DDoS-": "DDoS", "DoS-": "DoS", "Mirai-": "Mirai", "Recon-": "Recon"}


def to_group(label: str) -> str:
    """Map a raw CICIoT2023 label to one of GROUPS; raise on anything unknown."""
    label = label.strip()
    if label in _EXPLICIT:
        return _EXPLICIT[label]
    for prefix, group in _PREFIXES.items():
        if label.startswith(prefix):
            return group
    raise KeyError(f"unknown CICIoT2023 label: {label!r}")
