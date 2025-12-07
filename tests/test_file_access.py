"""
Tests for file_access module.
"""

import tempfile
import os
from pathlib import Path
from unittest.mock import patch
import pytest


# =============================================================================
# Tests for load_whitelist_raw
# =============================================================================

class TestLoadWhitelistRaw:
    """Tests for load_whitelist_raw function."""

    def test_empty_file_returns_empty_list(self, tmp_path):
        """Empty whitelist file should return empty list."""
        from sem_mem.file_access import load_whitelist_raw, WHITELIST_FILE

        whitelist_file = tmp_path / "sema_files.txt"
        whitelist_file.write_text("")

        with patch("sem_mem.file_access.WHITELIST_FILE", whitelist_file):
            result = load_whitelist_raw()

        assert result == []

    def test_comments_are_excluded(self, tmp_path):
        """Comment lines should not be included."""
        from sem_mem.file_access import load_whitelist_raw

        whitelist_file = tmp_path / "sema_files.txt"
        whitelist_file.write_text("""# This is a comment
sem_mem/core.py
# Another comment
app.py
""")

        with patch("sem_mem.file_access.WHITELIST_FILE", whitelist_file):
            result = load_whitelist_raw()

        assert result == ["sem_mem/core.py", "app.py"]

    def test_blank_lines_are_excluded(self, tmp_path):
        """Blank lines should not be included."""
        from sem_mem.file_access import load_whitelist_raw

        whitelist_file = tmp_path / "sema_files.txt"
        whitelist_file.write_text("""sem_mem/core.py

app.py

README.md
""")

        with patch("sem_mem.file_access.WHITELIST_FILE", whitelist_file):
            result = load_whitelist_raw()

        assert result == ["sem_mem/core.py", "app.py", "README.md"]

    def test_nonexistent_file_returns_empty_list(self, tmp_path):
        """Nonexistent whitelist file should return empty list."""
        from sem_mem.file_access import load_whitelist_raw

        with patch("sem_mem.file_access.WHITELIST_FILE", tmp_path / "nonexistent.txt"):
            result = load_whitelist_raw()

        assert result == []


# =============================================================================
# Tests for load_whitelist (expansion)
# =============================================================================

class TestLoadWhitelist:
    """Tests for load_whitelist function with path expansion."""

    def test_direct_file_entry(self, tmp_path):
        """Direct file entries should be included."""
        from sem_mem.file_access import load_whitelist

        # Create test structure
        (tmp_path / "test.py").write_text("# test")
        whitelist_file = tmp_path / "sema_files.txt"
        whitelist_file.write_text("test.py\n")

        with patch("sem_mem.file_access.BASE_DIR", tmp_path):
            with patch("sem_mem.file_access.WHITELIST_FILE", whitelist_file):
                result = load_whitelist()

        assert "test.py" in result

    def test_directory_expands_to_files(self, tmp_path):
        """Directory entries should expand to all contained files."""
        from sem_mem.file_access import load_whitelist

        # Create test structure
        subdir = tmp_path / "mydir"
        subdir.mkdir()
        (subdir / "file1.py").write_text("# file1")
        (subdir / "file2.py").write_text("# file2")

        whitelist_file = tmp_path / "sema_files.txt"
        whitelist_file.write_text("mydir/\n")

        with patch("sem_mem.file_access.BASE_DIR", tmp_path):
            with patch("sem_mem.file_access.WHITELIST_FILE", whitelist_file):
                result = load_whitelist()

        assert "mydir/file1.py" in result
        assert "mydir/file2.py" in result

    def test_directory_expands_recursively(self, tmp_path):
        """Directory entries should expand recursively."""
        from sem_mem.file_access import load_whitelist

        # Create nested structure
        subdir = tmp_path / "mydir"
        nested = subdir / "nested"
        nested.mkdir(parents=True)
        (subdir / "top.py").write_text("# top")
        (nested / "deep.py").write_text("# deep")

        whitelist_file = tmp_path / "sema_files.txt"
        whitelist_file.write_text("mydir/\n")

        with patch("sem_mem.file_access.BASE_DIR", tmp_path):
            with patch("sem_mem.file_access.WHITELIST_FILE", whitelist_file):
                result = load_whitelist()

        assert "mydir/top.py" in result
        assert "mydir/nested/deep.py" in result

    def test_excludes_hidden_directories(self, tmp_path):
        """Hidden directories should be excluded."""
        from sem_mem.file_access import load_whitelist

        # Create structure with hidden dir
        subdir = tmp_path / "mydir"
        hidden = subdir / ".hidden"
        hidden.mkdir(parents=True)
        (subdir / "visible.py").write_text("# visible")
        (hidden / "secret.py").write_text("# secret")

        whitelist_file = tmp_path / "sema_files.txt"
        whitelist_file.write_text("mydir/\n")

        with patch("sem_mem.file_access.BASE_DIR", tmp_path):
            with patch("sem_mem.file_access.WHITELIST_FILE", whitelist_file):
                result = load_whitelist()

        assert "mydir/visible.py" in result
        assert "mydir/.hidden/secret.py" not in result

    def test_excludes_pycache(self, tmp_path):
        """__pycache__ directories should be excluded."""
        from sem_mem.file_access import load_whitelist

        # Create structure with __pycache__
        subdir = tmp_path / "mydir"
        cache = subdir / "__pycache__"
        cache.mkdir(parents=True)
        (subdir / "module.py").write_text("# module")
        (cache / "module.cpython-311.pyc").write_bytes(b"compiled")

        whitelist_file = tmp_path / "sema_files.txt"
        whitelist_file.write_text("mydir/\n")

        with patch("sem_mem.file_access.BASE_DIR", tmp_path):
            with patch("sem_mem.file_access.WHITELIST_FILE", whitelist_file):
                result = load_whitelist()

        assert "mydir/module.py" in result
        assert not any("__pycache__" in p for p in result)

    def test_excludes_binary_artifacts(self, tmp_path):
        """Binary artifacts (.pyc, .so, etc.) should be excluded."""
        from sem_mem.file_access import load_whitelist

        subdir = tmp_path / "mydir"
        subdir.mkdir()
        (subdir / "module.py").write_text("# module")
        (subdir / "module.pyc").write_bytes(b"compiled")
        (subdir / "lib.so").write_bytes(b"shared object")

        whitelist_file = tmp_path / "sema_files.txt"
        whitelist_file.write_text("mydir/\n")

        with patch("sem_mem.file_access.BASE_DIR", tmp_path):
            with patch("sem_mem.file_access.WHITELIST_FILE", whitelist_file):
                result = load_whitelist()

        assert "mydir/module.py" in result
        assert "mydir/module.pyc" not in result
        assert "mydir/lib.so" not in result

    def test_includes_pdf_files(self, tmp_path):
        """PDF files should be included."""
        from sem_mem.file_access import load_whitelist

        subdir = tmp_path / "docs"
        subdir.mkdir()
        (subdir / "guide.pdf").write_bytes(b"%PDF-1.4")
        (subdir / "notes.md").write_text("# notes")

        whitelist_file = tmp_path / "sema_files.txt"
        whitelist_file.write_text("docs/\n")

        with patch("sem_mem.file_access.BASE_DIR", tmp_path):
            with patch("sem_mem.file_access.WHITELIST_FILE", whitelist_file):
                result = load_whitelist()

        assert "docs/guide.pdf" in result
        assert "docs/notes.md" in result

    def test_includes_word_docs(self, tmp_path):
        """Word documents should be included."""
        from sem_mem.file_access import load_whitelist

        subdir = tmp_path / "docs"
        subdir.mkdir()
        (subdir / "report.docx").write_bytes(b"PK")  # docx is a zip
        (subdir / "old.doc").write_bytes(b"\xd0\xcf")  # old doc format

        whitelist_file = tmp_path / "sema_files.txt"
        whitelist_file.write_text("docs/\n")

        with patch("sem_mem.file_access.BASE_DIR", tmp_path):
            with patch("sem_mem.file_access.WHITELIST_FILE", whitelist_file):
                result = load_whitelist()

        assert "docs/report.docx" in result
        assert "docs/old.doc" in result

    def test_prevents_directory_traversal(self, tmp_path):
        """Directory traversal attempts should be blocked."""
        from sem_mem.file_access import load_whitelist

        # Create a file outside the base dir
        outside = tmp_path.parent / "outside.txt"
        if outside.parent.exists():
            outside.write_text("secret")

        whitelist_file = tmp_path / "sema_files.txt"
        whitelist_file.write_text("../outside.txt\n")

        with patch("sem_mem.file_access.BASE_DIR", tmp_path):
            with patch("sem_mem.file_access.WHITELIST_FILE", whitelist_file):
                result = load_whitelist()

        # The traversal entry should be ignored
        assert not any("outside" in p for p in result)


# =============================================================================
# Tests for validate_file_access
# =============================================================================

class TestValidateFileAccess:
    """Tests for validate_file_access function."""

    def test_valid_whitelisted_file(self, tmp_path):
        """Valid whitelisted file should return its path."""
        from sem_mem.file_access import validate_file_access

        test_file = tmp_path / "test.py"
        test_file.write_text("# test")

        allowed = {"test.py"}

        with patch("sem_mem.file_access.BASE_DIR", tmp_path):
            result = validate_file_access("test.py", allowed)

        assert result == test_file

    def test_nonexistent_file_raises(self, tmp_path):
        """Nonexistent file should raise FileNotFoundError."""
        from sem_mem.file_access import validate_file_access

        allowed = {"nonexistent.py"}

        with patch("sem_mem.file_access.BASE_DIR", tmp_path):
            with pytest.raises(FileNotFoundError):
                validate_file_access("nonexistent.py", allowed)

    def test_non_whitelisted_file_raises(self, tmp_path):
        """Non-whitelisted file should raise PermissionError."""
        from sem_mem.file_access import validate_file_access

        test_file = tmp_path / "secret.py"
        test_file.write_text("# secret")

        allowed = {"other.py"}  # secret.py is not allowed

        with patch("sem_mem.file_access.BASE_DIR", tmp_path):
            with pytest.raises(PermissionError):
                validate_file_access("secret.py", allowed)

    def test_directory_traversal_raises(self, tmp_path):
        """Directory traversal attempts should raise PermissionError."""
        from sem_mem.file_access import validate_file_access

        allowed = {"safe.py"}

        with patch("sem_mem.file_access.BASE_DIR", tmp_path):
            with pytest.raises(PermissionError):
                validate_file_access("../../../etc/passwd", allowed)

    def test_normalizes_path(self, tmp_path):
        """Paths with different formats should be normalized."""
        from sem_mem.file_access import validate_file_access

        subdir = tmp_path / "sub"
        subdir.mkdir()
        test_file = subdir / "test.py"
        test_file.write_text("# test")

        allowed = {"sub/test.py"}

        with patch("sem_mem.file_access.BASE_DIR", tmp_path):
            # Both formats should work
            result1 = validate_file_access("sub/test.py", allowed)
            result2 = validate_file_access("sub\\test.py", allowed)

            assert result1 == test_file
            assert result2 == test_file


# =============================================================================
# Tests for get_file_info
# =============================================================================

class TestGetFileInfo:
    """Tests for get_file_info function."""

    def test_returns_file_list_with_metadata(self, tmp_path):
        """Should return list with path, size, and content_type."""
        from sem_mem.file_access import get_file_info

        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello')")

        allowed = {"test.py"}

        with patch("sem_mem.file_access.BASE_DIR", tmp_path):
            result = get_file_info(allowed)

        assert len(result) == 1
        assert result[0]["path"] == "test.py"
        assert result[0]["size"] > 0
        assert result[0]["content_type"] == "text"

    def test_identifies_pdf_content_type(self, tmp_path):
        """PDF files should have content_type 'pdf'."""
        from sem_mem.file_access import get_file_info

        test_file = tmp_path / "doc.pdf"
        test_file.write_bytes(b"%PDF-1.4")

        allowed = {"doc.pdf"}

        with patch("sem_mem.file_access.BASE_DIR", tmp_path):
            result = get_file_info(allowed)

        assert result[0]["content_type"] == "pdf"

    def test_identifies_word_content_type(self, tmp_path):
        """Word files should have content_type 'word'."""
        from sem_mem.file_access import get_file_info

        docx = tmp_path / "doc.docx"
        docx.write_bytes(b"PK")

        doc = tmp_path / "old.doc"
        doc.write_bytes(b"\xd0\xcf")

        allowed = {"doc.docx", "old.doc"}

        with patch("sem_mem.file_access.BASE_DIR", tmp_path):
            result = get_file_info(allowed)

        types = {r["path"]: r["content_type"] for r in result}
        assert types["doc.docx"] == "word"
        assert types["old.doc"] == "word"

    def test_skips_nonexistent_files(self, tmp_path):
        """Nonexistent files in whitelist should be skipped."""
        from sem_mem.file_access import get_file_info

        test_file = tmp_path / "exists.py"
        test_file.write_text("# exists")

        allowed = {"exists.py", "missing.py"}

        with patch("sem_mem.file_access.BASE_DIR", tmp_path):
            result = get_file_info(allowed)

        assert len(result) == 1
        assert result[0]["path"] == "exists.py"


# =============================================================================
# Tests for read_file_content
# =============================================================================

class TestReadFileContent:
    """Tests for read_file_content function."""

    def test_reads_text_file(self, tmp_path):
        """Text files should be read as text."""
        from sem_mem.file_access import read_file_content

        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello')")

        content, media_type = read_file_content(test_file)

        assert content == "print('hello')"
        assert "text/plain" in media_type

    def test_reads_pdf_as_bytes(self, tmp_path):
        """PDF files should be read as bytes."""
        from sem_mem.file_access import read_file_content

        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4 content")

        content, media_type = read_file_content(test_file)

        assert isinstance(content, bytes)
        assert content == b"%PDF-1.4 content"
        assert media_type == "application/pdf"

    def test_reads_docx_as_bytes(self, tmp_path):
        """DOCX files should be read as bytes."""
        from sem_mem.file_access import read_file_content

        test_file = tmp_path / "test.docx"
        test_file.write_bytes(b"PK\x03\x04...")

        content, media_type = read_file_content(test_file)

        assert isinstance(content, bytes)
        assert "wordprocessingml" in media_type

    def test_reads_doc_as_bytes(self, tmp_path):
        """DOC files should be read as bytes."""
        from sem_mem.file_access import read_file_content

        test_file = tmp_path / "test.doc"
        test_file.write_bytes(b"\xd0\xcf\x11\xe0...")

        content, media_type = read_file_content(test_file)

        assert isinstance(content, bytes)
        assert media_type == "application/msword"


# =============================================================================
# Tests for whitelist management
# =============================================================================

class TestWhitelistManagement:
    """Tests for add_to_whitelist and remove_from_whitelist."""

    def test_add_new_entry(self, tmp_path):
        """Adding a new entry should append to file."""
        from sem_mem.file_access import add_to_whitelist, load_whitelist_raw

        whitelist_file = tmp_path / "sema_files.txt"
        whitelist_file.write_text("existing.py\n")

        with patch("sem_mem.file_access.WHITELIST_FILE", whitelist_file):
            result = add_to_whitelist("new.py")
            entries = load_whitelist_raw()

        assert result is True
        assert "new.py" in entries
        assert "existing.py" in entries

    def test_add_duplicate_returns_false(self, tmp_path):
        """Adding a duplicate entry should return False."""
        from sem_mem.file_access import add_to_whitelist

        whitelist_file = tmp_path / "sema_files.txt"
        whitelist_file.write_text("existing.py\n")

        with patch("sem_mem.file_access.WHITELIST_FILE", whitelist_file):
            result = add_to_whitelist("existing.py")

        assert result is False

    def test_remove_existing_entry(self, tmp_path):
        """Removing an existing entry should remove it from file."""
        from sem_mem.file_access import remove_from_whitelist, load_whitelist_raw

        whitelist_file = tmp_path / "sema_files.txt"
        whitelist_file.write_text("keep.py\nremove.py\nalso_keep.py\n")

        with patch("sem_mem.file_access.WHITELIST_FILE", whitelist_file):
            result = remove_from_whitelist("remove.py")
            entries = load_whitelist_raw()

        assert result is True
        assert "remove.py" not in entries
        assert "keep.py" in entries
        assert "also_keep.py" in entries

    def test_remove_nonexistent_returns_false(self, tmp_path):
        """Removing a nonexistent entry should return False."""
        from sem_mem.file_access import remove_from_whitelist

        whitelist_file = tmp_path / "sema_files.txt"
        whitelist_file.write_text("existing.py\n")

        with patch("sem_mem.file_access.WHITELIST_FILE", whitelist_file):
            result = remove_from_whitelist("nonexistent.py")

        assert result is False

    def test_preserves_comments(self, tmp_path):
        """Removing entries should preserve comments."""
        from sem_mem.file_access import remove_from_whitelist

        whitelist_file = tmp_path / "sema_files.txt"
        whitelist_file.write_text("# Important comment\nkeep.py\nremove.py\n")

        with patch("sem_mem.file_access.WHITELIST_FILE", whitelist_file):
            remove_from_whitelist("remove.py")
            content = whitelist_file.read_text()

        assert "# Important comment" in content
        assert "keep.py" in content
        assert "remove.py" not in content
