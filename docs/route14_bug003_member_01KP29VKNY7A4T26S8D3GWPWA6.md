# Route14 BUG-003 Fix

## Scope
- Bug: `BUG-003`
- Owner: `01KP29VKNY7A4T26S8D3GWPWA6`
- Source: `docs/bug_fix_list.md`

## Changes
- Ensured config schema responses expose runtime vs inferred source explicitly.
- Added explicit save-block reason for inferred schema responses.
- Preserved structured-save guard so unloaded plugins cannot use inferred schema as runtime truth.

## Files
- `src/webui/routers/plugin.py`
- `docs/bug_fix_list.md`

## Verification
- Read config schema route branches for runtime and inferred outputs.
- Confirm structured save path rejects unloaded plugin instances.
- Confirm response now exposes save-block metadata for inferred schema consumers.

## History Relation
- Direct fix alignment for `BUG-003`.
- Related to prior DTO/schema observability findings, but no new bug added.
