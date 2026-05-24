import sys
sys.path.insert(0, '/app')

exec(open('/app/code.py').read())

result = simulate_lock_pick(num_pins=6, has_security_pins=True, seed=42)

assert isinstance(result, dict), "Result should be a dict"
assert 'binding_order' in result, "Must have binding_order"
assert 'picks_needed' in result, "Must have picks_needed"
assert 'false_sets_encountered' in result, "Must have false_sets_encountered"
assert 'lock_opened' in result, "Must have lock_opened"
assert 'pin_states' in result, "Must have pin_states"

assert result['lock_opened'] == True, f"Lock should be opened, got {result['lock_opened']}"

bo = result['binding_order']
assert len(bo) == 6, f"Binding order should have 6 entries, got {len(bo)}"
assert len(set(bo)) == 6, f"Binding order should have 6 unique indices, got {bo}"
assert all(0 <= idx <= 5 for idx in bo), f"All indices should be 0-5, got {bo}"

assert result['picks_needed'] >= 6, f"picks_needed should be >= 6, got {result['picks_needed']}"
assert result['false_sets_encountered'] > 0, f"false_sets should be > 0 with security pins, got {result['false_sets_encountered']}"

ps = result['pin_states']
assert len(ps) == 6, f"pin_states should have 6 entries, got {len(ps)}"
for pin in ps:
    assert pin['set'] == True, f"Pin {pin.get('index', '?')} should be set, got {pin}"
    assert 'height' in pin, "pin state must have height"
    assert 'tolerance' in pin, "pin state must have tolerance"

result2 = simulate_lock_pick(num_pins=6, has_security_pins=True, seed=42)
assert result2['binding_order'] == result['binding_order'], "Results should be deterministic with same seed"
assert result2['picks_needed'] == result['picks_needed'], "Results should be deterministic with same seed"

result3 = simulate_lock_pick(num_pins=6, has_security_pins=False, seed=42)
assert result3['false_sets_encountered'] == 0, f"No security pins should mean 0 false sets, got {result3['false_sets_encountered']}"
assert result3['lock_opened'] == True, "Lock should still open without security pins"

print("PASS")

