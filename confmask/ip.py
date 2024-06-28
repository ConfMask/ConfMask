"""
TODO
"""

_used_ips = set()


def generate_unicast_ip(rng, b0=None, b1=None, b2=None, b3=None):
    """Generate a randpm unicast IP.

    Parameters
    ----------
    rng : Generator
        The numpy random generator instance.
    b1, b2, b3, b4 : int or tuple, optional
        The bytes of the IP address, randomly generated if not given. If the given bytes
        cannot satisfy unicast requirement, raise an error.

    Returns
    -------
    bytes : tuple of int
        The generated IP address bytes.
    """

    def _next():
        byte0 = b0 or rng.integers(1, 224)
        assert 1 <= byte0 < 224
        byte1 = b1 or rng.integers(0, 256)
        assert 0 <= byte1 < 256
        byte2 = b2 or rng.integers(0, 256)
        assert 0 <= byte2 < 256
        byte3 = b3 or rng.integers(0, 256)
        assert 0 <= byte3 < 256
        return (byte0, byte1, byte2, byte3)

    s = _next()
    while s in _used_ips:
        s = _next()
    _used_ips.add(s)
    return s
