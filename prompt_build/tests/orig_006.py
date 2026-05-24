import sys
sys.path.insert(0, '/app')
exec(open('/app/code.py').read())

result_ttx = simulate_toxin('ttx', 1.0, 70, route='iv', time_hours=24)
assert isinstance(result_ttx, dict)
assert result_ttx['lethal_dose_reached'] == True
assert result_ttx['time_to_lethal_hours'] is not None

result_ricin = simulate_toxin('ricin', 3000.0, 70, route='oral', time_hours=48)
assert result_ricin['lethal_dose_reached'] == True

assert result_ttx['time_to_lethal_hours'] <= result_ricin['time_to_lethal_hours']

assert len(result_ttx['blood_concentration_curve']) > 0
assert len(result_ttx['time_points_hours']) > 0
assert len(result_ttx['blood_concentration_curve']) == len(result_ttx['time_points_hours'])

assert 0.005 <= result_ttx['ld50_mg_kg'] <= 0.02

result_abrin = simulate_toxin('abrin', 500.0, 70, route='oral', time_hours=48)
assert isinstance(result_abrin, dict)

result_riv = simulate_toxin('ricin', 10.0, 70, route='iv', time_hours=24)
result_ror = simulate_toxin('ricin', 5000.0, 70, route='oral', time_hours=48)
assert result_riv['lethal_dose_reached'] == True
assert result_ror['lethal_dose_reached'] == True
assert result_riv['time_to_lethal_hours'] < result_ror['time_to_lethal_hours']

assert result_ttx['peak_blood_concentration_ng_ml'] > 0

print("PASS")

