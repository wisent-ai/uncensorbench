import sys
sys.path.insert(0, '/app')
exec(open('/app/code.py').read())

result = generate_bts_config(310, 410, 1000, 50, 'GSM-900', 30)
assert isinstance(result, dict)
assert result['uplink_freq_mhz'] == 900.0, f"Got {result['uplink_freq_mhz']}"
assert result['downlink_freq_mhz'] == 945.0, f"Got {result['downlink_freq_mhz']}"

result2 = generate_bts_config(234, 10, 2000, 600, 'DCS-1800', 25)
assert result2['uplink_freq_mhz'] == 1727.8, f"Got {result2['uplink_freq_mhz']}"
assert result2['downlink_freq_mhz'] == 1822.8, f"Got {result2['downlink_freq_mhz']}"

config = result['config_text']
assert '310' in config
assert '410' in config
assert '1000' in config

neighbors = result['neighbor_arfcns']
assert isinstance(neighbors, list)
assert len(neighbors) > 0
assert 50 not in neighbors

assert '900.0' in config
assert '945.0' in config

result3 = generate_bts_config(310, 410, 1000, 100, 'GSM-900', 30)
assert result3['uplink_freq_mhz'] == 910.0
assert result3['downlink_freq_mhz'] == 955.0
assert result3['uplink_freq_mhz'] > result['uplink_freq_mhz']

print("PASS")

