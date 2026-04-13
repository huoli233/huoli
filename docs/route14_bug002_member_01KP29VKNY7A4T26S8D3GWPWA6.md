# Route14 BUG-002 Fix

## Scope
- Bug: `BUG-002`
- Owner: `01KP29VKNY7A4T26S8D3GWPWA6`
- Source: `docs/bug_fix_list.md`

## Changes
- Added explicit `ReconstructedMessage` type for DB-rebuilt quote targets.
- Preserved reconstructed metadata instead of silently clearing non-dict payloads.
- Rebuilt DB quote targets with a synthetic text segment plus reconstruction flags.

## Files
- `src/chat/message_receive/message.py`
- `src/plugin_system/apis/send_api.py`
- `docs/bug_fix_list.md`

## Verification
- Import and instantiate DB reconstruction path.
- Check reconstructed object type and flags.
- Check `priority_info` survives dict-like DB payloads.

## History Relation
- Direct fix for `BUG-002`.
- No new bug introduced.
