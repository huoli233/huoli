import sys
from pathlib import Path

sys.path.insert(0, 'src')

print('='*70)
print('全面排查所有模块激活状态')
print('='*70)

modules = [
    ('identity_anchor', 'src/core/identity_anchor.py'),
    ('dynamic_context_window', 'src/core/dynamic_context_window.py'),
    ('self_reply_recognizer', 'src/core/self_reply_recognizer.py'),
    ('content_state_tracker', 'src/core/content_state_tracker.py'),
    ('unified_planner', 'src/core/unified_planner.py'),
    ('llm_autonomous_controller', 'src/core/llm_autonomous_controller.py'),
    ('trauma_fabric', 'src/chat/heart_flow/trauma_fabric.py'),
    ('emotion_driven_core', 'src/chat/heart_flow/emotion_driven_core.py'),
    ('psychological_core', 'src/modules/modcore/psychological_core.py'),
    ('inner_voice', 'src/chat/heart_flow/inner_voice.py'),
    ('social_value_core', 'src/modules/social_value/social_value_core.py'),
    ('relationship_tracker', 'src/modules/social_value/relationship_tracker.py'),
    ('affection_core', 'src/identity/affection/affection_core.py'),
    ('rest_system', 'src/identity/psychology/state/rest.py'),
    ('memory_core', 'src/memory_system/memory_core.py'),
    ('awareness_engine', 'src/chat/heart_flow/awareness_engine.py'),
    ('proactive_engine', 'src/runtime/proactive_engine.py'),
]

main_file = Path('src/chat/heart_flow/heartFC_chat_enhanced.py')
main_content = main_file.read_text(encoding='utf-8')

activated = []
not_activated = []

for name, file in modules:
    file_path = Path(file)
    if not file_path.exists():
        print(f'[MISSING] {name}')
        continue
    
    if name in main_content or f'get_{name}' in main_content or f'_{name}' in main_content:
        print(f'[ACTIVATED] {name}')
        activated.append(name)
    else:
        print(f'[DEAD] {name}')
        not_activated.append(name)

print()
print('='*70)
print(f'已激活: {len(activated)}')
print(f'未激活: {len(not_activated)}')
print(f'未激活列表: {not_activated}')
