# subject_area_utils.py
import re

def parse_topic_area(item: str):
    """
    item like: 'Machine learning -> Time-series analysis'
    returns (topic, area)
    """
    if not item or not isinstance(item, str):
        return None, None
    parts = [p.strip() for p in item.split("->", 1)]
    if len(parts) != 2:
        return None, None
    topic, area = parts[0], parts[1]
    return topic, area

def parse_secondary_list(s: str):
    """
    secondary field like:
    'Climate change -> X; Machine learning -> Y; ...'
    returns list of (topic, area)
    """
    if not s or not isinstance(s, str):
        return []
    items = [x.strip() for x in s.split(";") if x.strip()]
    out = []
    for it in items:
        t, a = parse_topic_area(it)
        if t and a:
            out.append((t, a))
    return out

def normalize_topic(t: str):
    if not t:
        return ""
    t = t.strip().lower()
    # keep canonical spellings
    if "machine" in t and "learning" in t:
        return "machine_learning"
    if "climate" in t and "change" in t:
        return "climate_change"
    return re.sub(r"\s+", "_", t)
