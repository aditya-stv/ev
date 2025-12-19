"""
Cleanup Script - Remove Temporary and Unwanted Files
====================================================
Removes all temporary scripts, old visualizations, and documentation
while keeping core code, best model, and final visualizations.
"""

import os
import shutil
from pathlib import Path

print("="*80)
print("CLEANING UP TEMPORARY AND UNWANTED FILES")
print("="*80)

# Files and folders to KEEP
KEEP_FILES = {
    # Core training/model files
    'main_final_ensemble.py',
    'deep_svdd_advanced.py',
    'contractive_ae.py',
    'advanced_features.py',
    
    # Production files
    'amtead_production.py',
    'test_production.py',
    
    # Configuration
    'requirements.txt',
    
    # This cleanup script itself
    'cleanup_project.py'
}

KEEP_FOLDERS = {
    'models',           # Best trained model
    'models_best',      # Backup of best model
    'new_visualization' # Final IEEE-quality visualizations
}

# Get all files and folders in current directory
all_items = os.listdir('.')

print("\nItems to REMOVE:")
print("-"*80)

removed_count = 0
kept_count = 0

# Remove unwanted Python scripts
for item in all_items:
    item_path = Path(item)
    
    # Skip if it's one we want to keep
    if item in KEEP_FILES or item in KEEP_FOLDERS:
        kept_count += 1
        continue
    
    # Skip hidden files and __pycache__
    if item.startswith('.') or item == '__pycache__':
        continue
    
    try:
        if item_path.is_file():
            # Remove temporary Python scripts
            if item.endswith('.py'):
                print(f"  Removing script: {item}")
                os.remove(item)
                removed_count += 1
            
            # Remove markdown documentation
            elif item.endswith('.md'):
                print(f"  Removing doc: {item}")
                os.remove(item)
                removed_count += 1
            
            # Remove text files
            elif item.endswith('.txt') and item != 'requirements.txt':
                print(f"  Removing txt: {item}")
                os.remove(item)
                removed_count += 1
                
        elif item_path.is_dir():
            # Remove old visualization folders
            if 'visualization' in item.lower() and item != 'new_visualization':
                print(f"  Removing folder: {item}/")
                shutil.rmtree(item)
                removed_count += 1
            
            # Remove old model backup folders
            elif item.startswith('models_run'):
                print(f"  Removing folder: {item}/")
                shutil.rmtree(item)
                removed_count += 1
                
    except Exception as e:
        print(f"  ⚠ Error removing {item}: {e}")

print("-"*80)
print(f"\nCleaned up: {removed_count} items")
print(f"Kept: {kept_count} essential items")

print("\n" + "="*80)
print("FINAL PROJECT STRUCTURE")
print("="*80)

print("\n📁 Essential Code Files:")
for f in sorted(KEEP_FILES):
    if os.path.exists(f):
        size = os.path.getsize(f) / 1024
        print(f"  ✓ {f:<40} ({size:.1f} KB)")

print("\n📁 Essential Folders:")
for folder in sorted(KEEP_FOLDERS):
    if os.path.exists(folder):
        folder_path = Path(folder)
        file_count = len(list(folder_path.rglob('*')))
        folder_size = sum(f.stat().st_size for f in folder_path.rglob('*') if f.is_file())
        print(f"  ✓ {folder + '/':<40} ({file_count} files, {folder_size/1024/1024:.1f} MB)")

print("\n" + "="*80)
print("✅ CLEANUP COMPLETE!")
print("="*80)
print("\nProject is now clean and ready for production/publication.")
print("="*80)
