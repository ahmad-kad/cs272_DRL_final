# Locked Files

This file documents files that are locked/protected in this repository to prevent accidental modifications.

## Locked Files

### `environments/urban_junction_env.py`
- **Status**: LOCKED (Read-only)
- **Reason**: This is the core Urban Junction Environment implementation that has been thoroughly tested and validated. Changes to this file could break the entire training pipeline.
- **Protection**: File is marked as read-only (`attrib +r`)

## How to Temporarily Unlock

If you need to modify a locked file (e.g., for bug fixes or enhancements):

1. Make the file writable:
   ```bash
   attrib -r environments/urban_junction_env.py
   ```

2. Make your changes

3. Test thoroughly - run the full training pipeline to ensure nothing is broken

4. Restore read-only protection:
   ```bash
   attrib +r environments/urban_junction_env.py
   ```

5. Commit your changes with a clear explanation of why the modification was necessary

## Why Lock Files?

- **Stability**: Prevents accidental changes that could break working code
- **Quality Assurance**: Ensures core components are only modified intentionally
- **Safety**: Protects validated implementations from unintended modifications

## Requesting Changes

If you need to modify a locked file:
1. Create an issue describing the proposed change
2. Explain why the change is necessary
3. Get approval from the project maintainer
4. Follow the unlock procedure above</contents>
</xai:function_call">The file was created successfully.