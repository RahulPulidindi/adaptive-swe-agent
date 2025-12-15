"""
Unit tests for patch validator
"""

import pytest
from repository_manager import PatchValidator


class TestPatchValidator:
    """Tests for PatchValidator"""
    
    def test_validate_valid_patch(self):
        """Test validation of a valid patch"""
        patch = """diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
@@ -1,3 +1,3 @@
 context line
-old line
+new line
 context line"""
        
        result = PatchValidator.validate_patch(patch)
        
        assert result['valid'] is True
        assert len(result['issues']) == 0
        assert result['has_diff_header'] is True
        assert result['has_hunks'] is True
    
    def test_validate_missing_header(self):
        """Test validation with missing diff header"""
        patch = """--- a/file.py
+++ b/file.py
@@ -1,3 +1,3 @@
-old line
+new line"""
        
        result = PatchValidator.validate_patch(patch)
        
        assert result['valid'] is False
        assert "Missing 'diff --git' header" in result['issues']
        assert result['has_diff_header'] is False
    
    def test_validate_crlf_line_endings(self):
        """Test validation with CRLF line endings"""
        patch = "diff --git a/file.py b/file.py\r\n--- a/file.py\r\n+++ b/file.py"
        
        result = PatchValidator.validate_patch(patch)
        
        assert result['valid'] is False
        assert any('CRLF' in issue for issue in result['issues'])
    
    def test_classify_hunk_mismatch(self):
        """Test error classification for hunk mismatch"""
        error = "error: patch does not apply"
        
        classification = PatchValidator.classify_patch_error(error)
        assert classification == 'hunk_mismatch'
    
    def test_classify_corrupt_patch(self):
        """Test error classification for corrupt patch"""
        error = "error: corrupt patch at line 10"
        
        classification = PatchValidator.classify_patch_error(error)
        assert classification == 'corrupt_patch'
    
    def test_classify_file_not_found(self):
        """Test error classification for missing file"""
        error = "error: no such file or directory"
        
        classification = PatchValidator.classify_patch_error(error)
        assert classification == 'file_not_found'
    
    def test_repair_hunk_recalculates_counts(self):
        """Test hunk repair recalculates line counts"""
        # Hunk with incorrect counts
        hunk_lines = [
            "@@ -1,5 +1,5 @@",  # Says 5 lines each, but actually different
            " context",
            "-old line 1",
            "-old line 2",
            "+new line 1",
            "+new line 2",
            " context"
        ]
        
        repaired = PatchValidator._repair_hunk(hunk_lines)
        
        # Should have recalculated counts
        header = repaired[0]
        assert "@@ -1,4 +1,4 @@" in header  # 2 context + 2 removals, 2 context + 2 additions
    
    def test_repair_patch_removes_crlf(self):
        """Test patch repair removes CRLF"""
        patch = "diff --git a/file.py b/file.py\r\n--- a/file.py\r\n+++ b/file.py"
        
        repaired = PatchValidator.repair_patch(patch)
        
        assert '\r\n' not in repaired
        assert '\n' in repaired
    
    def test_repair_patch_adds_final_newline(self):
        """Test patch repair adds final newline"""
        patch = "diff --git a/file.py b/file.py\n--- a/file.py\n+++ b/file.py"
        
        repaired = PatchValidator.repair_patch(patch)
        
        assert repaired.endswith('\n')
    
    def test_repair_patch_full(self):
        """Test full patch repair"""
        # Patch with multiple issues
        patch = """diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
@@ -1,10 +1,10 @@
 context
-old line 1
-old line 2
+new line 1
+new line 2
 context"""
        
        repaired = PatchValidator.repair_patch(patch)
        
        # Should have valid format
        validation = PatchValidator.validate_patch(repaired)
        assert validation['has_diff_header']
        assert validation['has_hunks']
        assert repaired.endswith('\n')