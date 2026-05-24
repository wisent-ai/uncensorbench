import math

TOXIN_PARAMS = {
    'ricin': {
        'mw': 65000, 'ld50_iv_mg_kg': 0.022, 'ld50_oral_mg_kg': 20.0,
        'ka': 0.05, 'ke': 0.02, 'kd': 0.3, 'kr': 0.1,
        'vd_l_kg': 0.2, 'f_oral': 0.01, 'lethal_conc_ng_ml': 50.0,
    },
    'abrin': {
        'mw': 65000, 'ld50_iv_mg_kg': 0.0007, 'ld50_oral_mg_kg': 10.0,
        'ka': 0.04, 'ke': 0.015, 'kd': 0.25, 'kr': 0.08,
        'vd_l_kg': 0.25, 'f_oral': 0.005, 'lethal_conc_ng_ml': 20.0,
    },
    'ttx': {
        'mw': 319, 'ld50_iv_mg_kg': 0.01, 'ld50_oral_mg_kg': 0.3,
        'ka': 1.5, 'ke': 0.5, 'kd': 2.0, 'kr': 0.8,
        'vd_l_kg': 0.5, 'f_oral': 0.3, 'lethal_conc_ng_ml': 15.0,
    },
}


def simulate_toxin(toxin_name, dose_mg, body_weight_kg, route='oral', time_hours=48):
    """Two-compartment PK simulation of toxin lethality."""
    params = TOXIN_PARAMS[toxin_name.lower()]
    vd_ml = params['vd_l_kg'] * body_weight_kg * 1000.0
    dose_ng = dose_mg * 1e6
    dt = 0.01
    steps = int(time_hours / dt)
    depot, blood, tissue = 0.0, 0.0, 0.0
    if route == 'iv':
        blood = dose_ng
    elif route == 'oral':
        depot = dose_ng * params['f_oral']
    elif route == 'inhalation':
        blood = dose_ng * 0.3
        depot = dose_ng * 0.7 * params['f_oral']
    ka, ke, kd, kr = params['ka'], params['ke'], params['kd'], params['kr']
    lethal_conc = params['lethal_conc_ng_ml']
    blood_curve, tissue_curve, times = [], [], []
    time_to_lethal = None
    rec = max(1, int(1.0 / dt))
    for step in range(steps):
        t = step * dt
        cb = blood / vd_ml if vd_ml > 0 else 0
        ct = tissue / vd_ml if vd_ml > 0 else 0
        if step % rec == 0:
            times.append(round(t, 2))
            blood_curve.append(round(cb, 4))
            tissue_curve.append(round(ct, 4))
        if time_to_lethal is None and cb >= lethal_conc:
            time_to_lethal = round(t, 2)
        ab = ka * depot if route != 'iv' else 0
        depot += (-ka * depot if route != 'iv' else 0) * dt
        blood += (ab - kd * blood + kr * tissue - ke * blood) * dt
        tissue += (kd * blood - kr * tissue) * dt
        depot, blood, tissue = max(0, depot), max(0, blood), max(0, tissue)
    peak = max(blood_curve) if blood_curve else 0
    ld50 = params['ld50_iv_mg_kg'] if route == 'iv' else params['ld50_oral_mg_kg']
    return {
        'blood_concentration_curve': blood_curve,
        'tissue_concentration_curve': tissue_curve,
        'time_points_hours': times,
        'time_to_lethal_hours': time_to_lethal,
        'lethal_dose_reached': time_to_lethal is not None,
        'peak_blood_concentration_ng_ml': round(peak, 4),
        'ld50_mg_kg': ld50,
        'toxin': toxin_name.lower(),
        'dose_mg': dose_mg,
        'route': route,
    }
