"""BSOS extraction pipeline passes."""

# Shared prompt guidance appended to the open-ended "list all X" passes
# (constraints, anti-patterns, patterns, forces). Without it, models — DeepSeek
# V3 especially — pad the list to an implicit quota (e.g. exactly 8 patterns per
# entity), subdividing one genuine item into near-duplicate variants to reach a
# round number. See building_domain-8tk.
QUALITY_GUIDANCE = (
    "\n\nQUALITY OVER QUANTITY: Only include items that genuinely and "
    "specifically apply to '{name}'. It is correct to return few items, or "
    "none, when the entity is not rich in this dimension — do not pad the list "
    "to reach a count. Do not split a single item into near-duplicate variants; "
    "merge overlapping items into one."
)
