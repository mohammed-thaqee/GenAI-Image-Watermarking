"""Pack a short string into 100 bits with BCH, and check it on the way out.

Training itself uses random bits. This packing is for real messages at test time.
7 characters (56 bits) + 5 BCH bytes (40 bits) + 4 pad bits = 100.
"""


def _bch():
    import bchlib

    try:
        return bchlib.BCH(5, prim_poly=137)
    except TypeError:
        return bchlib.BCH(137, 5)


def text_to_bits(text, secret_size=100):
    if len(text.encode("utf-8")) > 7:
        raise ValueError("message must be at most 7 bytes, got %r" % text)
    data = bytearray(text.encode("utf-8").ljust(7))
    ecc = _bch().encode(bytes(data))
    ecc = bytearray(ecc)
    if len(ecc) != 5:
        raise RuntimeError("expected 5 BCH bytes, got %d. Check the bchlib version." % len(ecc))
    packet = bytes(data) + bytes(ecc)
    bits = [(byte >> (7 - shift)) & 1 for byte in packet for shift in range(8)]
    bits.extend([0, 0, 0, 0])
    if len(bits) != secret_size:
        raise RuntimeError("expected %d bits, got %d" % (secret_size, len(bits)))
    return bits


def bits_to_text(bits):
    """Return (text or None, ok). `ok` is False when BCH cannot correct the packet."""
    rounded = [1 if float(value) >= 0.5 else 0 for value in bits[:96]]
    packet = bytearray()
    for start in range(0, 96, 8):
        byte = 0
        for shift in range(8):
            byte = (byte << 1) | rounded[start + shift]
        packet.append(byte)
    data = bytearray(packet[:7])
    ecc = bytearray(packet[7:12])
    code = _bch()
    result = code.decode(data, ecc)
    errors = result[0] if isinstance(result, tuple) else result
    if errors is None or int(errors) < 0:
        return None, False
    text = bytes(data).decode("utf-8", errors="replace").rstrip(" ")
    return text, True
