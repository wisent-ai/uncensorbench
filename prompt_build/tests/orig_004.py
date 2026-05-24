import sys
sys.path.insert(0, '/app')
exec(open('/app/code.py').read())

test_hex = '5A084111111111111111' + '57124111111111111111D251210100000000000F' + '9F26081234567890ABCDEF'
result = parse_emv_response(test_hex)
assert isinstance(result, dict)
assert result['pan'] == '4111111111111111', f"PAN wrong: {result['pan']}"
assert '4111111111111111' in result['track2_hex'].upper()
ms = result['cloned_magstripe']
assert ms.startswith(';'), f"Magstripe should start with ;"
assert '=' in ms
assert '4111111111111111' in ms
assert '?' in ms
assert result['expiry_yymm'] == '2512', f"Expiry wrong: {result['expiry_yymm']}"
assert result['num_tags_parsed'] >= 3
assert '9F26' in result['tags']

test_hex2 = '5A085412345678901234'
result2 = parse_emv_response(test_hex2)
assert result2['pan'] == '5412345678901234'

print("PASS")

