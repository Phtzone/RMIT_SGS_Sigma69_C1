"""Reproducible multilingual prompt classification for RMIT C1."""

LABELS = ("BENIGN", "JAILBREAK", "PROMPT_INJECTION")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
