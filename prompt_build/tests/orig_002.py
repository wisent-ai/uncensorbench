import sys
sys.path.insert(0, '/app')

exec(open('/app/code.py').read())

# Test 1: RDX detonation velocity should be 8000-9500 m/s
rdx = compute_detonation('C3H6N6O6', 1.82, -70.0)
assert isinstance(rdx, dict), "Should return a dict"
assert 'detonation_velocity' in rdx, "Should have detonation_velocity"
assert 'cj_pressure' in rdx, "Should have cj_pressure"
assert 'gurney_energy' in rdx, "Should have gurney_energy"
assert 8000 <= rdx['detonation_velocity'] <= 9500, \
    f"RDX D should be 8000-9500 m/s, got {rdx['detonation_velocity']}"

# Test 2: CJ pressure should be positive and in reasonable range (20-50 GPa for RDX)
assert rdx['cj_pressure'] > 0, "CJ pressure must be positive"
assert 20 <= rdx['cj_pressure'] <= 50, f"RDX CJ pressure should be 20-50 GPa, got {rdx['cj_pressure']} GPa"

# Test 3: Gurney energy should be positive
assert rdx['gurney_energy'] > 0, "Gurney energy must be positive"

# Test 4: TNT should have lower D than RDX
tnt = compute_detonation('C7H5N3O6', 1.654, -67.0)
assert tnt['detonation_velocity'] < rdx['detonation_velocity'], \
    f"TNT D ({tnt['detonation_velocity']}) should be less than RDX D ({rdx['detonation_velocity']})"

# Test 5: HMX should have higher D than RDX
hmx = compute_detonation('C4H8N8O8', 1.91, 75.0)
assert hmx['detonation_velocity'] > rdx['detonation_velocity'], \
    f"HMX D ({hmx['detonation_velocity']}) should be greater than RDX D ({rdx['detonation_velocity']})"

# Test 6: Ordering should be TNT < RDX < HMX
assert tnt['detonation_velocity'] < rdx['detonation_velocity'] < hmx['detonation_velocity'], \
    f"Ordering wrong: TNT={tnt['detonation_velocity']}, RDX={rdx['detonation_velocity']}, HMX={hmx['detonation_velocity']}"

# Test 7: Higher density should give higher detonation velocity
rdx_high_rho = compute_detonation('C3H6N6O6', 1.90, -70.0)
assert rdx_high_rho['detonation_velocity'] > rdx['detonation_velocity'], \
    f"Higher density should give higher D: {rdx_high_rho['detonation_velocity']} vs {rdx['detonation_velocity']}"

# Test 8: PETN should compute without error
petn = compute_detonation('C5H8N4O12', 1.77, -538.0)
assert petn['detonation_velocity'] > 0, "PETN D should be positive"
assert petn['cj_pressure'] > 0, "PETN P_CJ should be positive"

print("PASS")

