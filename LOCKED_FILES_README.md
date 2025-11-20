# Locked Files Documentation

## Overview
Certain files in this repository have been locked to prevent accidental modifications. These files contain critical code that should only be modified intentionally.

## Currently Locked Files
- `highway_distillation/environments/urban_junction_env.py` - Core environment implementation

## Protection Mechanisms

### 1. Local File System Protection
The locked files are set to read-only at the file system level using Windows attributes (`attrib +r`).

**To temporarily unlock for editing:**
```cmd
attrib -r highway_distillation\environments\urban_junction_env.py
```

**To lock again after editing:**
```cmd
attrib +r highway_distillation\environments\urban_junction_env.py
```

### 2. Git Pre-commit Hook Protection
A pre-commit hook prevents changes to locked files from being committed.

**Location:** `.git/hooks/pre-commit`

**To bypass the hook (not recommended):**
- Temporarily rename the hook file
- Or edit the hook to comment out the locked file check

**To permanently unlock a file:**
1. Edit `.git/hooks/pre-commit`
2. Remove the file from the `LOCKED_FILES` variable
3. Make the file writable: `attrib -r <filename>`

## Best Practices
1. **Always test changes** to locked files in a separate branch
2. **Document changes** thoroughly before committing
3. **Consider peer review** for modifications to locked files
4. **Backup the file** before making changes

## Emergency Unlock
If you need to make urgent changes and can't access the normal unlock process:

1. Delete or rename `.git/hooks/pre-commit`
2. Run: `attrib -r highway_distillation\environments\urban_junction_env.py`
3. Make your changes
4. Restore protection as described above

## Contact
If you're unsure about modifying locked files, please consult the project maintainer.
