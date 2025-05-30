import re


def extract_items(text: str, section: str):
    pattern = rf"{section}:\s*\((.*?)\)"
    match = re.search(pattern, text)
    if match:
        return [item.strip() for item in match.group(1).split(",")]
    return []
