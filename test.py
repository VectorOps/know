"""
A small tool to calculate token counts for files in a project using litellm.

This script walks the current directory, finds files with specific extensions,
skips files and directories listed in .gitignore, and then calculates
the token count for each file using a specified model.

Usage:
1. Make sure you have litellm installed:
   pip install litellm openai

2. Run the script from the root of your project:
   python test.py
"""

import os
import fnmatch
import litellm

# --- Configuration ---
# The user requested 'openai/o3'. If this model name doesn't work,
# you might want to try "gpt-4o", "gpt-3.5-turbo", or another model supported by litellm.
MODEL_NAME = "openai/o3" 

# File extensions to include in the token count
TARGET_EXTENSIONS = {".py", ".js", ".ts", ".go", ".md", ".txt"}

# Path to the .gitignore file
GITIGNORE_PATH = ".gitignore"

def get_gitignore_patterns(gitignore_path):
    """
    Reads patterns from a .gitignore file and splits them into directory-only
    patterns (those ending with '/') and general patterns.
    """
    dir_only_patterns = []
    general_patterns = []
    if not os.path.exists(gitignore_path):
        return dir_only_patterns, general_patterns
    
    with open(gitignore_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                if line.endswith('/'):
                    dir_only_patterns.append(line.rstrip('/'))
                else:
                    general_patterns.append(line)
    return dir_only_patterns, general_patterns

def main():
    """
    Main function to walk through the project, filter files,
    and calculate token counts.
    """
    dir_only_patterns, general_patterns = get_gitignore_patterns(GITIGNORE_PATH)
    total_tokens = 0
    processed_files = 0

    print(f"Starting token count for extensions: {', '.join(TARGET_EXTENSIONS)}")
    print(f"Using model: {MODEL_NAME}")
    print("-" * 60)

    this_script_name = "test.py"

    for root, dirs, files in os.walk(".", topdown=True):
        # Prune ignored directories from traversal.
        # A general .gitignore pattern (without '/') matches files and directories.
        # A directory-only pattern (with '/') matches only directories.
        all_dir_patterns = dir_only_patterns + general_patterns
        dirs[:] = [d for d in dirs if not any(fnmatch.fnmatch(d, p) for p in all_dir_patterns)]
        
        for filename in files:
            filepath = os.path.join(root, filename)
            # Normalize path for comparison to avoid issues with './'
            normalized_filepath = os.path.normpath(filepath)

            # Avoid processing this script itself
            if normalized_filepath == this_script_name:
                continue

            # Check file extension
            _, ext = os.path.splitext(filename)
            if ext not in TARGET_EXTENSIONS:
                continue

            # Check if filename is ignored by any general pattern.
            # Directory-only patterns don't apply to files.
            if any(fnmatch.fnmatch(filename, p) for p in general_patterns):
                continue
            
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                
                token_count = litellm.token_counter(model=MODEL_NAME, text=content)
                print(f"{filepath:<50} | Tokens: {token_count}")
                total_tokens += token_count
                processed_files += 1

            except Exception as e:
                print(f"Error processing {filepath}: {e}")

    print("-" * 60)
    print(f"Processed {processed_files} files.")
    print(f"Total tokens: {total_tokens}")

if __name__ == "__main__":
    main()
