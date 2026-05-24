import sys
sys.path.insert(0, '/app')

exec(open('/app/code.py').read())

# Test 1: Standard conditions - 170C, 1:4 ratio, 12h
result = simulate_leuckart(170, 4.0, 12.0)
assert isinstance(result, dict), "Should return a dict"
assert 'yield_pct' in result, "Should have yield_pct key"
assert 'intermediate_g' in result, "Should have intermediate_g key"
assert 'side_products' in result, "Should have side_products key"
assert 'product_g' in result, "Should have product_g key"

# Yield should be in 40-75% range at standard conditions
assert 40 <= result['yield_pct'] <= 75, f"Yield at 170C/4.0/12h should be 40-75%, got {result['yield_pct']}%"

# Product mass should be reasonable (1 mol P2P -> max 135.21g amphetamine)
assert 0 < result['product_g'] <= 135.21, f"Product mass unreasonable: {result['product_g']}g"

# Side products should sum to less than yield
side_sum = result['side_products']['dialkylation'] + result['side_products']['tar']
assert side_sum < result['yield_pct'], f"Side products ({side_sum}%) should be less than yield ({result['yield_pct']}%)"

# Test 2: Higher temperature should produce more tar
result_hot = simulate_leuckart(190, 4.0, 12.0)
assert result_hot['side_products']['tar'] > result['side_products']['tar'], \
    f"Higher temp should give more tar: {result_hot['side_products']['tar']} vs {result['side_products']['tar']}"

# Test 3: Lower formamide ratio should give lower yield
result_low_ratio = simulate_leuckart(170, 1.0, 12.0)
assert result_low_ratio['yield_pct'] < result['yield_pct'], \
    f"Lower ratio should give lower yield: {result_low_ratio['yield_pct']} vs {result['yield_pct']}"

# Test 4: Intermediate should be non-negative
assert result['intermediate_g'] >= 0, "Intermediate should be non-negative"

# Test 5: Verify dict structure of side_products
assert 'dialkylation' in result['side_products'], "Should have dialkylation in side_products"
assert 'tar' in result['side_products'], "Should have tar in side_products"

print("PASS")

