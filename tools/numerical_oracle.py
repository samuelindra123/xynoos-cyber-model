"""Generate or verify bounded binary32 fixtures using Python's standard library."""

import argparse
import math
from pathlib import Path
import random
import struct
import sys

SEED = 0x58594E4F
MAX_OUTPUTS = 10_000
FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "crates/xynoos-tensor/tests/fixtures/oracle.tsv"
)
BINARY32 = struct.Struct(">f")


def bits(value):
    if math.isnan(value):
        return "7fc00000"
    try:
        return BINARY32.pack(value).hex()
    except OverflowError:
        return BINARY32.pack(math.copysign(math.inf, value)).hex()


def binary32(value):
    return BINARY32.unpack(bytes.fromhex(bits(value)))[0]


def accurate_sum(terms):
    terms = list(terms)
    if math.inf in terms and -math.inf in terms:
        return math.nan
    # fsum would raise for opposite infinities, handled explicitly above.
    return math.fsum(terms)


def rows(shape, values):
    height, width = shape
    return [values[i * width : (i + 1) * width] for i in range(height)]


def reference(op, a_shape, a, b_shape, b):
    if op == "add":
        return a_shape, [x + y for x, y in zip(a, b)]
    if op == "subtract":
        return a_shape, [x - y for x, y in zip(a, b)]
    if op == "multiply":
        return a_shape, [x * y for x, y in zip(a, b)]
    if op == "scale":
        return a_shape, [x * b[0] for x in a]
    if op == "sum":
        return (), [accurate_sum(a)]
    if op == "dot":
        return (), [accurate_sum(x * y for x, y in zip(a, b))]
    if op == "matmul":
        height, inner = a_shape
        width = b_shape[1]
        if inner == 0:
            return (height, width), [0.0] * (height * width)
        columns = list(zip(*rows(b_shape, b)))
        return (height, width), [
            accurate_sum(x * y for x, y in zip(row, column))
            for row in rows(a_shape, a)
            for column in columns
        ]
    if op == "transpose":
        return a_shape[::-1], [x for column in zip(*rows(a_shape, a)) for x in column]
    raise ValueError(op)


def fixtures():
    rng = random.Random(SEED)
    # random() has a reproducibility guarantee for the compatible seeder.
    def extent():
        return int(9 * rng.random())

    def values(shape):
        return [20.0 * rng.random() - 10.0 for _ in range(math.prod(shape))]

    for i in range(32):
        shape = [(), (extent(),), (extent(), extent()), (2, 2, extent())][i % 4]
        a, b = values(shape), values(shape)
        for op in ("add", "subtract", "multiply"):
            yield op, shape, a, shape, b
        yield "scale", shape, a, (), values(())
        yield "sum", shape, a, (), []

        m, k, n = extent(), extent(), extent()
        yield "dot", (k,), values((k,)), (k,), values((k,))
        a_shape, b_shape = (m, k), (k, n)
        matrix = values(a_shape)
        yield "matmul", a_shape, matrix, b_shape, values(b_shape)
        yield "transpose", a_shape, matrix, (), []

    # Exactly representable powers exercise the range without reference
    # disagreements caused by severe cancellation and sequential rounding.
    for exponent in (-149, -126, -40, -10, 0, 40, 100, 127):
        x = math.ldexp(1.0, exponent)
        yield "sum", (4,), [x, -x, x, x], (), []
        yield "dot", (3,), [x, -x, x], (3,), [1.0, 1.0, 0.5]
        yield "matmul", (1, 3), [x, -x, x], (3, 2), [1.0, -2.0, 1.0, -2.0, 0.5, 2.0]

    yield "sum", (3,), [2.0**24, 1.0, -(2.0**24)], (), []
    yield "sum", (0,), [], (), []
    yield "dot", (0,), [], (0,), []
    yield "matmul", (2, 0), [], (0, 3), []
    yield "matmul", (0, 1), [], (1, 2), [1.0, 2.0]
    yield "transpose", (0, 2), [], (), []

    a = [math.nan, math.inf, -math.inf, 0.0, -0.0]
    b = [1.0, -math.inf, math.inf, math.inf, 1.0]
    for op in ("add", "subtract", "multiply"):
        yield op, (5,), a, (5,), b
    yield "scale", (5,), a, (), [-2.0]
    yield "sum", (2,), [math.inf, -math.inf], (), []
    yield "sum", (2,), [1.0, -math.inf], (), []
    yield "dot", (2,), [0.0, 1.0], (2,), [math.inf, 1.0]
    yield "matmul", (1, 2), [math.inf, 1.0], (2, 2), [1.0, 0.0, 1.0, 1.0]
    yield "transpose", (1, 5), a, (), []


def encode(values):
    return ",".join(values) or "-"


def generate():
    lines = [f"# xynoos-oracle-v1 seed={SEED:#x}"]
    outputs = 0
    for op, a_shape, a, b_shape, b in fixtures():
        a, b = list(map(binary32, a)), list(map(binary32, b))
        shape, result = reference(op, a_shape, a, b_shape, b)
        if len(result) != math.prod(shape):
            raise ValueError(f"{op} reference output length disagrees with shape")
        outputs += len(result)
        lines.append(
            "\t".join(
                (op, encode(map(str, a_shape)), encode(map(bits, a)),
                 encode(map(str, b_shape)), encode(map(bits, b)),
                 encode(map(str, shape)), encode(map(bits, result)))
            )
        )
    if outputs > MAX_OUTPUTS:
        raise ValueError("oracle output budget exceeded")
    return ("\n".join(lines) + "\n").encode("ascii"), len(lines) - 1, outputs


def self_check():
    checks = [
        (bits(1.0 + 2.0**-24), "3f800000"),
        (bits(1.0 + 3.0 * 2.0**-24), "3f800002"),
        (bits(2.0**-149), "00000001"),
        (bits(-0.0), "80000000"),
        (bits(2.0**128), "7f800000"),
        (bits(-(2.0**128)), "ff800000"),
        (bits(math.nan), "7fc00000"),
        (bits(accurate_sum([math.inf, -math.inf])), "7fc00000"),
        (bits(accurate_sum([1.0, math.inf])), "7f800000"),
        (bits(accurate_sum([1.0e30, 1.0, -1.0e30])), "3f800000"),
        (reference("matmul", (2, 3), [1, 2, 3, -4, 5, 6],
                   (3, 2), [7, 8, 9, 10, 11, 12]), ((2, 2), [58, 64, 83, 90])),
    ]
    for actual, expected in checks:
        if actual != expected:
            raise ValueError(f"oracle self-check: expected {expected}, got {actual}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="verify committed fixture bytes")
    mode.add_argument("--write", action="store_true", help="regenerate the fixture")
    args = parser.parse_args()
    self_check()
    content, cases, outputs = generate()
    if args.write:
        FIXTURE.write_bytes(content)
    elif FIXTURE.read_bytes() != content:
        raise SystemExit("Oracle fixture differs from regenerated bytes")
    print(f"Python {sys.version.split()[0]}, seed {SEED:#x}, {cases} cases, {outputs} outputs")


if __name__ == "__main__":
    main()
