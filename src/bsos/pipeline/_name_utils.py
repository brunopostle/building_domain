"""Shared name-normalisation utilities for pass name-lookup tables."""


def normalized_forms(name: str) -> list[str]:
    """Return alternative lowercase forms of name for plural/singular matching.

    Only generates forms in the direction that's missing — if the name ends
    in 's' we try stripping it; if it doesn't we try adding it. This keeps
    the number of spurious matches low.
    """
    forms: list[str] = []
    if name.endswith("ies") and len(name) > 4:
        forms.append(name[:-3] + "y")          # activities → activity
    elif name.endswith("ses") and len(name) > 4:
        forms.append(name[:-2])                 # processes → process
    elif name.endswith("es") and len(name) > 4:
        forms.append(name[:-1])                 # cores → core (via -s rule below too)
        forms.append(name[:-2])                 # processes → process
    elif name.endswith("s") and not name.endswith("ss") and len(name) > 3:
        forms.append(name[:-1])                 # piles → pile, materials → material
    else:
        forms.append(name + "s")                # pile → piles
        if name.endswith("y") and len(name) > 2:
            forms.append(name[:-1] + "ies")     # activity → activities
    return forms
