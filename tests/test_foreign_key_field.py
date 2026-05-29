"""Tests for lightweight foreign-key field metadata."""

import pytest
from lancedb.pydantic import LanceModel
from pydantic import ValidationError

from homeobox.schema import ForeignKeyField


class AuthorSchema(LanceModel):
    uid: str


class PublisherSchema(LanceModel):
    publisher_id: str


class BookSchema(LanceModel):
    author_uid: str = ForeignKeyField.declare(target_schema=AuthorSchema)
    publisher_id: str | None = ForeignKeyField.declare(
        target_schema="PublisherSchema",
        target_field="publisher_id",
    )
    title: str


def test_foreign_key_field_metadata_is_in_json_schema_extra():
    extra = BookSchema.model_fields["author_uid"].json_schema_extra

    assert extra == {
        "foreign_key": {
            "target_schema": "AuthorSchema",
            "target_field": "uid",
        }
    }


def test_foreign_key_field_does_not_enforce_referential_integrity():
    book = BookSchema(author_uid="missing-author", publisher_id="missing-publisher", title="T")

    assert book.author_uid == "missing-author"
    assert book.publisher_id == "missing-publisher"


def test_foreign_key_field_remains_a_regular_required_pydantic_field():
    with pytest.raises(ValidationError, match="author_uid"):
        BookSchema(publisher_id=None, title="T")


def test_foreign_key_field_is_not_written_to_arrow_metadata():
    schema = BookSchema.to_arrow_schema()

    assert schema.field("author_uid").metadata is None
    assert schema.field("publisher_id").metadata is None
