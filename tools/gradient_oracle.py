"""Independent binary64 gradient and learning references for TASK 003."""

import argparse
import hashlib
from pathlib import Path
import struct

FIXTURES = Path(__file__).resolve().parents[1] / "crates/xynoos-autograd/tests/fixtures"
POLYNOMIAL = (-0.375,)
VECTOR = (0.75, -0.5, 1.25, -1.0, 0.5, 2.0)
TARGET = (0.25, -0.75, 1.5)
MATRIX = (
    0.5, -1.0, 2.0, -0.75, 1.5, 0.25,
    1.0, -2.0, 0.5, 0.25, 0.75, 0.5, -1.0, 2.0, -0.5, 1.25, 1.5, -0.75,
)
UPSTREAM = ((0.5, -1.25, 2.0, 0.75), (-1.5, 0.25, 0.5, -2.0))


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def polynomial(values):
    x, = values
    return (3 * x + 2) ** 2 + x * x + x


def vector(values):
    a, b = values[:3], values[3:]
    return -1.5 * sum((x * y + x - t) ** 2 for x, y, t in zip(a, b, TARGET)) / 3


def matrix(values):
    a, b = values[:6], values[6:]
    return sum(
        UPSTREAM[i][j] * sum(a[3 * i + q] * b[4 * q + j] for q in range(3))
        for i in range(2) for j in range(4)
    )


def analytical_matrix():
    a = [MATRIX[i:i + 3] for i in (0, 3)]
    b = [MATRIX[i:i + 4] for i in (6, 10, 14)]
    # Apply the product rule to each term of the weighted scalar objective.
    left = [0.0] * 6
    right = [0.0] * 12
    for row, weights in enumerate(UPSTREAM):
        for column, weight in enumerate(weights):
            for inner, factor in enumerate(a[row]):
                left[3 * row + inner] += weight * b[inner][column]
                right[4 * inner + column] += weight * factor
    return left + right


def central(values, index, evaluate):
    x = values[index]
    h = 1e-3 * max(1.0, abs(x))
    hi, lo = list(values), list(values)
    hi[index], lo[index] = f32(x + h), f32(x - h)
    return (evaluate(hi) - evaluate(lo)) / (hi[index] - lo[index])


def gradient_rows():
    x, = POLYNOMIAL
    a, b = VECTOR[:3], VECTOR[3:]
    d = [-(u * v + u - t) for u, v, t in zip(a, b, TARGET)]
    vector_exact = [g * (1 + v) for g, v in zip(d, b)] + [g * u for g, u in zip(d, a)]
    cases = [
        ("polynomial", POLYNOMIAL, polynomial, [6 * (3 * x + 2) + 2 * x + 1]),
        ("vector", VECTOR, vector, vector_exact),
        ("matrix", MATRIX, matrix, analytical_matrix()),
    ]
    rows = []
    for name, values, evaluate, exact in cases:
        for i, derivative in enumerate(exact):
            numerical = central(values, i, evaluate)
            if abs(derivative - numerical) > 2e-6 + 2e-5 * max(abs(derivative), abs(numerical)):
                raise ValueError(f"independent gradient cross-check failed for {name}[{i}]")
            rows.append(f"{name} {i} {derivative:.17g} {numerical:.17g}\n")
    return rows


def learning_rows():
    errors = (-2.0, -3.0)
    rows = []
    for step in range(101):
        w, b = errors
        loss = 2 * w * w + b * b
        rows.append(f"{step} {w + 2:.17g} {b + 3:.17g} {loss:.17g}\n")
        errors = 0.8 * w, 0.9 * b
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args()

    for filename, rows, count in [
        ("gradients.txt", gradient_rows(), 25),
        ("learning.txt", learning_rows(), 101),
    ]:
        if len(rows) != count:
            raise SystemExit(f"{filename}: expected {count} rows, got {len(rows)}")
        data = "".join(rows).encode("ascii")
        path = FIXTURES / filename
        if args.write:
            path.write_bytes(data)
        elif path.read_bytes() != data:
            raise SystemExit(f"{filename}: reference fixture differs")
        print(f"{filename}: {count} rows, SHA-256 {hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
