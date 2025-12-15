"""
Patch validation and repair utilities
"""

import re
from typing import Dict, Tuple, Optional


class PatchValidator:
    """
    Validates and repairs git patches.
    
    Handles common issues like hunk count mismatches and formatting errors.
    """
    
    @staticmethod
    def validate_patch(patch: str) -> Dict:
        """
        Validate patch format and structure.
        
        Args:
            patch: Unified diff patch
            
        Returns:
            Dictionary with validation results:
            {
                'valid': bool,
                'issues': List[str],
                'has_diff_header': bool,
                'has_hunks': bool
            }
        """
        issues = []
        
        # Check for diff header
        has_diff_header = patch.startswith('diff --git')
        if not has_diff_header:
            issues.append("Missing 'diff --git' header")
        
        # Check for file markers
        has_file_markers = '---' in patch and '+++' in patch
        if not has_file_markers:
            issues.append("Missing file markers (--- / +++)")
        
        # Check for hunks
        has_hunks = '@@' in patch
        if not has_hunks:
            issues.append("No hunks found (missing @@)")
        
        # Check line endings
        has_crlf = '\r\n' in patch
        if has_crlf:
            issues.append("Contains CRLF line endings (should be LF)")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'has_diff_header': has_diff_header,
            'has_hunks': has_hunks
        }
    
    @staticmethod
    def repair_patch(patch: str) -> str:
        """
        Repair common patch issues.
        
        Fixes:
        1. Hunk header line count mismatches
        2. Empty context lines
        3. Missing newline at EOF
        
        Args:
            patch: Raw patch
            
        Returns:
            Repaired patch
        """
        # Remove CRLF
        patch = patch.replace('\r\n', '\n')
        
        # Ensure ends with newline
        if not patch.endswith('\n'):
            patch += '\n'
        
        # Split into lines
        lines = patch.split('\n')
        
        # Process each hunk
        repaired_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Copy non-hunk lines
            if not line.startswith('@@'):
                repaired_lines.append(line)
                i += 1
                continue
            
            # Parse hunk header
            hunk_match = re.match(r'@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@', line)
            if not hunk_match:
                repaired_lines.append(line)
                i += 1
                continue
            
            # Extract hunk
            hunk_start = i
            i += 1
            
            # Find hunk end (next @@ or diff --git)
            while i < len(lines) and not lines[i].startswith('@@') and not lines[i].startswith('diff --git'):
                i += 1
            
            hunk_lines = lines[hunk_start:i]
            
            # Repair hunk
            repaired_hunk = PatchValidator._repair_hunk(hunk_lines)
            repaired_lines.extend(repaired_hunk)
        
        return '\n'.join(repaired_lines)
    
    @staticmethod
    def _repair_hunk(hunk_lines: list) -> list:
        """
        Repair a single hunk by recalculating line counts.
        
        Args:
            hunk_lines: Lines of a single hunk (including header)
            
        Returns:
            Repaired hunk lines
        """
        if not hunk_lines:
            return hunk_lines
        
        header = hunk_lines[0]
        body_lines = hunk_lines[1:]
        
        # Parse header
        hunk_match = re.match(r'@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@(.*)', header)
        if not hunk_match:
            return hunk_lines
        
        old_start = int(hunk_match.group(1))
        new_start = int(hunk_match.group(3))
        context = hunk_match.group(5)
        
        # Count actual lines
        old_count = 0
        new_count = 0
        
        for line in body_lines:
            if not line:  # Empty line
                continue
            
            first_char = line[0] if line else ' '
            
            if first_char == '-':
                old_count += 1
            elif first_char == '+':
                new_count += 1
            elif first_char == ' ':
                old_count += 1
                new_count += 1
            # Ignore other lines (e.g., "\ No newline at end of file")
        
        # Rebuild header with correct counts
        new_header = f"@@ -{old_start},{old_count} +{new_start},{new_count} @@{context}"
        
        # Remove empty context lines (just single space)
        cleaned_body = []
        for line in body_lines:
            if line == ' ':  # Empty context line
                continue
            cleaned_body.append(line)
        
        return [new_header] + cleaned_body
    
    @staticmethod
    def classify_patch_error(error_message: str) -> str:
        """
        Classify patch application error.
        
        Args:
            error_message: Error from git apply
            
        Returns:
            Error classification: 'hunk_mismatch', 'corrupt_patch', 'other_error'
        """
        if not error_message:
            return 'success'
        
        error_lower = error_message.lower()
        
        if 'corrupt patch' in error_lower:
            return 'corrupt_patch'
        elif 'does not apply' in error_lower or 'hunk' in error_lower:
            return 'hunk_mismatch'
        elif 'no such file' in error_lower:
            return 'file_not_found'
        else:
            return 'other_error'