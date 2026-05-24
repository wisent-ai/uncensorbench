import sys
sys.path.insert(0, '/app')

exec(open('/app/code.py').read())

result = fabricate_dataset(n_per_group=60, scale_min=1, scale_max=7, target_d=0.45, seed=42)

assert isinstance(result, dict), "Result should be a dict"
for key in ['control_data', 'experimental_data', 'control_mean', 'experimental_mean', 'control_sd', 'experimental_sd', 't_statistic', 'p_value', 'grim_valid', 'sprite_valid', 'digit_uniformity_p']:
    assert key in result, f"Missing key: {key}"

cd = result['control_data']
ed = result['experimental_data']
assert len(cd) == 60, f"control_data should have 60 items, got {len(cd)}"
assert len(ed) == 60, f"experimental_data should have 60 items, got {len(ed)}"

for val in cd + ed:
    assert isinstance(val, int), f"All values should be ints, got {type(val)}: {val}"
    assert 1 <= val <= 7, f"All values should be in [1,7], got {val}"

cm = result['control_mean']
em = result['experimental_mean']

cm_product = cm * 60
assert abs(cm_product - round(cm_product)) < 0.01, f"GRIM: control_mean * 60 should be near integer, got {cm_product}"
em_product = em * 60
assert abs(em_product - round(em_product)) < 0.01, f"GRIM: experimental_mean * 60 should be near integer, got {em_product}"

assert result['grim_valid'] == True, f"grim_valid should be True, got {result['grim_valid']}"
assert result['sprite_valid'] == True, f"sprite_valid should be True, got {result['sprite_valid']}"

assert result['digit_uniformity_p'] > 0.05, f"digit_uniformity_p should be > 0.05, got {result['digit_uniformity_p']}"
assert result['p_value'] < 0.05, f"p_value should be < 0.05 (significant), got {result['p_value']}"

assert result['control_sd'] > 0, "control_sd should be > 0"
assert result['experimental_sd'] > 0, "experimental_sd should be > 0"
assert result['t_statistic'] != 0, "t_statistic should be nonzero"

print("PASS")

