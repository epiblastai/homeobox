"""Validate an atlas schema YAML IR by parsing it all the way into a live atlas.

The YAML intermediate representation is the authored artifact; this script proves
it round-trips into working homeobox schema classes. Five stages, each a harder
check than the last, and each a hard failure with an informative error:

1. **Load** — parse the YAML into a ``SchemaModel`` (``homeobox.schema.ir``).
   Catches unknown keys, malformed markers, and bad field/enum payloads.
2. **Registries** — check every ``ontology_aligned`` / ``cross_reference`` marker
   value against ``polycomb.registry`` (the same ``parse_ontology`` /
   ``parse_crossref`` resolution tooling uses). The IR carries these as bare
   strings, so this is the only place a typo or a wrong-cased authority is
   caught before harmonization.
3. **Emit** — generate ``schema.py`` source (``homeobox.schema.emit``), which
   runs the output through ``ast.parse``. A clean return guarantees valid Python.
4. **Exec** — execute the generated source so every class is defined. This is
   where ``HoxBaseSchema`` validates pointer fields against the registered
   feature-space specs and enum-typed fields resolve.
5. **Atlas** — build a throwaway ``RaggedAtlas`` from the obs, dataset, and
   derived registry schemas, exercising Arrow/Lance schema generation.

    python validate_schema_ir.py <schema.yaml> [--emit-to schema.py] [--skip-atlas]
"""

import argparse
import os
import tempfile

# Importing homeobox registers the builtin feature-space specs that pointer
# fields are validated against when the generated classes are defined.
import homeobox as hox
from homeobox.schema import emit
from homeobox.schema.ir import SchemaModel, load_yaml_file

from polycomb.registry import parse_crossref, parse_ontology


def check_registry_values(model: SchemaModel) -> None:
    """Check ontology / cross-reference marker values against polycomb.registry.

    ``ontology_aligned`` and ``cross_reference`` markers carry bare strings in the
    IR; resolution tooling later parses them with ``parse_ontology`` /
    ``parse_crossref``, which match by registry *value* (e.g. ``NCBITaxon``, not
    the member name ``NCBITAXON``). Validate every marker the same way so a wrong
    authority fails here rather than silently at harmonization time.
    """
    tables = [
        *model.obs_tables,
        *model.feature_registry_tables,
        *model.fk_registry_tables,
        *model.other_tables,
    ]
    if model.dataset_table is not None:
        tables.append(model.dataset_table)

    problems: list[str] = []
    for table in tables:
        for field in table.fields:
            ontology = field.markers.get("ontology_aligned")
            if ontology is not None:
                try:
                    parse_ontology(ontology["ontology_name"])
                except ValueError as exc:
                    problems.append(f"{table.name}.{field.name}: {exc}")
            crossref = field.markers.get("cross_reference")
            if crossref is not None:
                try:
                    parse_crossref(crossref["database_name"])
                except ValueError as exc:
                    problems.append(f"{table.name}.{field.name}: {exc}")

    if problems:
        raise ValueError(
            "marker values not present in polycomb.registry:\n  " + "\n  ".join(problems)
        )


def validate(yaml_path: str, emit_to: str | None = None, skip_atlas: bool = False) -> None:
    # 1. Load: YAML -> SchemaModel.
    model = load_yaml_file(yaml_path)
    print(
        f"loaded IR: {model.name} ({len(model.obs_tables)} obs table(s), "
        f"{len(model.enums)} enum(s))"
    )

    # 2. Registries: ontology / cross-reference marker values must be known.
    check_registry_values(model)
    print("registry marker values OK")

    # 3. Emit: SchemaModel -> schema.py source (ast.parse runs inside emit()).
    source = emit(model)
    print(f"emitted schema.py source ({len(source)} chars)")
    if emit_to is not None:
        with open(emit_to, "w", encoding="utf-8") as handle:
            handle.write(source)
        print(f"wrote generated schema to {emit_to}")

    # 4. Exec: define every class so pointer fields and enum references resolve.
    namespace: dict[str, object] = {}
    exec(compile(source, "<generated schema.py>", "exec"), namespace)
    table_names = [
        t.name
        for t in (
            *model.obs_tables,
            *model.feature_registry_tables,
            *model.fk_registry_tables,
            *model.other_tables,
        )
    ]
    if model.dataset_table is not None:
        table_names.append(model.dataset_table.name)
    print(f"executed generated source; defined {len(table_names)} table class(es)")

    if skip_atlas:
        print("\nIR parses into valid Python (atlas creation skipped).")
        return

    # 5. Atlas: build a throwaway atlas from the live classes.
    obs_schemas = {t.name: namespace[t.name] for t in model.obs_tables}
    dataset_schema = namespace[model.dataset_table.name] if model.dataset_table else None
    registry_schemas = {fs: namespace[name] for fs, name in model.registry_schemas().items()}
    with tempfile.TemporaryDirectory() as tmpdir:
        hox.create_or_open_atlas(
            atlas_path=tmpdir,
            obs_schemas=obs_schemas,
            dataset_table_name="datasets",
            dataset_schema=dataset_schema,
            registry_schemas=registry_schemas,
        )
    print("\nIR parses into valid Python and a temporary atlas was created.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("schema_yaml", help="Path to the atlas schema YAML IR")
    parser.add_argument(
        "--emit-to", default=None, help="Optional path to write the generated schema.py"
    )
    parser.add_argument(
        "--skip-atlas",
        action="store_true",
        help="Stop after exec; do not build a temporary atlas",
    )
    args = parser.parse_args(argv)
    validate(os.fspath(args.schema_yaml), emit_to=args.emit_to, skip_atlas=args.skip_atlas)


if __name__ == "__main__":
    main()
