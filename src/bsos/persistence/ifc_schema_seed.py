"""Deterministic enumeration of canonical IFC entity classes from the IFC schema.

ifc_class entities used to be invented by the LLM (Pass 1 bootstrap and Pass 12),
which hallucinated names ('IfcHydronicHeatingSystem', 'IfcHotel') and prose
pseudo-classes ('IFC Mechanical Equipment'). Seeding from the authoritative
schema via ifcopenshell makes the ifc_class layer canonical-by-construction:
every name corresponds to a real IFC type and is marked with the deterministic
source SCHEMA_SOURCE so reactive purges (building_domain-4se) never touch them.
"""
from __future__ import annotations

from dataclasses import dataclass

# Deterministic source_model marker for schema-seeded ifc_class entities.
# Distinguishes them from LLM-minted ones so cleanup never purges them.
SCHEMA_SOURCE = "ifc-schema"

# Schema versions seeded by default. IFC4 is the project minimum (_test.ifc is
# IFC4); IFC4X3 adds infrastructure classes and is a superset for shared names.
DEFAULT_SCHEMAS = ["IFC4"]


@dataclass(frozen=True)
class IFCClass:
    """A canonical IFC entity class enumerated from the schema."""

    name: str
    schemas: tuple[str, ...]
    supertype: str | None
    is_abstract: bool

    @property
    def description(self) -> str:
        parts = [f"IFC entity class (schema {'/'.join(self.schemas)})."]
        if self.supertype:
            parts.append(f"Supertype: {self.supertype}.")
        if self.is_abstract:
            parts.append("Abstract type (not directly instantiable).")
        return " ".join(parts)


def iter_ifc_classes(schemas: list[str] | None = None) -> list[IFCClass]:
    """Enumerate canonical IFC entity classes, unioned by name across schemas.

    A class defined in multiple schema versions appears once, recording every
    schema that declares it. The first schema in ``schemas`` provides the
    canonical supertype/abstractness. Only entity declarations are returned —
    defined types, enumerations and selects (IfcLengthMeasure, IfcWallTypeEnum)
    are excluded because they never appear as the class of an IFC instance.
    """
    from ifcopenshell import ifcopenshell_wrapper as w

    schemas = schemas or DEFAULT_SCHEMAS
    by_name: dict[str, IFCClass] = {}

    for schema_name in schemas:
        schema = w.schema_by_name(schema_name)
        for decl in schema.declarations():
            entity = decl.as_entity()
            if entity is None:
                continue
            name = entity.name()
            if name in by_name:
                existing = by_name[name]
                by_name[name] = IFCClass(
                    name=existing.name,
                    schemas=existing.schemas + (schema_name,),
                    supertype=existing.supertype,
                    is_abstract=existing.is_abstract,
                )
                continue
            supertype = entity.supertype()
            by_name[name] = IFCClass(
                name=name,
                schemas=(schema_name,),
                supertype=supertype.name() if supertype else None,
                is_abstract=entity.is_abstract(),
            )

    return sorted(by_name.values(), key=lambda c: c.name)
