"""
Tests for Doc usage documentation snippets.
"""
from sieves import Doc

def test_doc_usage():
    # --8<-- [start:doc-usage]
    from sieves import Doc

    # Create a document from text
    doc = Doc(text="This is a sample document.")

    # Create a document with metadata
    doc_with_meta = Doc(
        text="Document with metadata.",
        meta={"source": "manual", "priority": "high"}
    )

    # Documents can also have a URI
    doc_with_uri = Doc(uri="https://example.com/doc.pdf")
    # --8<-- [end:doc-usage]
    assert doc.text == "This is a sample document."
