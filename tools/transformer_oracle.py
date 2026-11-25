#!/usr/bin/env python3
"""Standalone TASK 005 binary32 fixtures, reverse equations and negative controls."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "crates/xynoos-transformer/tests/fixtures"
SEED = 0x58594E4F4F530005
TINY = dict(v=260, d=16, n=1, h=2, f=48, context=8)
SMALL = dict(v=292, d=32, n=4, h=4, f=96, context=64)
CANDIDATE = dict(v=292, d=96, n=8, h=4, f=256, context=128)
C2_COORDINATES = (
    ("embedding", 53 * 96 + 7),
    ("block.6.attention_gain", 71),
    ("block.2.q", 3011),
    ("block.6.k", 811),
    ("block.4.v", 5377),
    ("block.7.o", 1300),
    ("block.3.ffn_gain", 82),
    ("block.1.gate", 22033),
    ("block.5.value", 20123),
    ("block.6.down", 17003),
    ("block.0.q", 809),
    ("final_gain", 49),
)


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bits(values):
    return [struct.pack("<f", value).hex() for value in values]


def check_initialization_edge_cases():
    mask = (1 << 64) - 1
    state = (SEED + 0x9E3779B97F4A7C15) & mask
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
    z ^= z >> 31
    assert state == 0xF690C808CE9D7C1A
    assert z == 0x5DFD1045191BA975
    assert (z >> 11) == 0xBBFA208A32375
    lower = 0.5 / (1 << 53)
    upper = min((((1 << 53) - 1) + 0.5) / (1 << 53), math.nextafter(1.0, 0.0))
    assert lower == 2.0 ** -54
    assert upper == math.nextafter(1.0, 0.0)
    assert 0.0 < lower < upper < 1.0


def inventory(v, d, n, h, f, context):
    yield "embedding", (v, d)
    for layer in range(n):
        for name, shape in (
            ("attention_gain", (d,)), ("q", (d, d)), ("k", (d, d)),
            ("v", (d, d)), ("o", (d, d)), ("ffn_gain", (d,)),
            ("gate", (d, f)), ("value", (d, f)), ("down", (f, d)),
        ):
            yield f"block.{layer}.{name}", shape
    yield "final_gain", (d,)


def initialized(config):
    def uniform():
        state = SEED
        mask = (1 << 64) - 1
        while True:
            state = (state + 0x9E3779B97F4A7C15) & mask
            z = state
            z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
            z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
            z ^= z >> 31
            yield min(((z >> 11) + 0.5) / (1 << 53), math.nextafter(1, 0))

    stream = uniform()
    for name, shape in inventory(**config):
        count = math.prod(shape)
        if name == "embedding":
            values = []
            while len(values) < count:
                radius = math.sqrt(-2 * math.log(next(stream)))
                angle = 2 * math.pi * next(stream)
                values.extend(f32(0.02 * f32(radius * trig(angle))) for trig in (math.cos, math.sin))
            yield name, values[:count]
        elif len(shape) == 1:
            yield name, [1.0] * count
        else:
            bound = math.sqrt(6 / sum(shape))
            if name.endswith((".o", ".down")):
                bound *= 1 / math.sqrt(2 * config["n"])
            yield name, [f32((2 * next(stream) - 1) * bound) for _ in range(count)]


def explicit(config):
    def entries():
        for p, (name, shape) in enumerate(inventory(**config)):
            count = math.prod(shape)
            if len(shape) == 1:
                values = [1 + ((i + p) % 9 - 4) / 32 for i in range(count)]
            else:
                values = [(((i * 67 + i // 11 * 19 + p * 43) % 257) - 128) /
                          (2048 if p == 0 else 768) for i in range(count)]
            yield name, list(map(f32, values))
    return dict(entries())


class Graph:
    def __init__(self):
        self.nodes = []

    def result(self, values, *edges):
        node = len(self.nodes)
        self.nodes.append((list(map(f32, values)), edges))
        return node

    def value(self, node):
        return self.nodes[node][0]

    def backward(self, root):
        pending = {root: [1.0]}
        gradients = {}
        for node in reversed(range(root + 1)):
            if node not in pending:
                continue
            grad = list(map(f32, pending.pop(node)))
            gradients[node] = grad
            for parent, vjp in self.nodes[node][1]:
                dest = pending.setdefault(parent, [0.0] * len(self.value(parent)))
                for i, delta in enumerate(vjp(grad)):
                    dest[i] += f32(delta)
        return gradients

    def dense(self, a, w, d, n, tied=False):
        x, raw = self.value(a), self.value(w)
        rows = len(x) // d
        weights = list(zip(*(raw[j:j+d] for j in range(0, len(raw), d)))) if tied else [
            raw[j:j+n] for j in range(0, len(raw), n)]
        xrows = [x[i:i+d] for i in range(0, len(x), d)]
        columns = list(zip(*weights))

        def dx(g):
            for i in range(rows):
                for weight_row in weights:
                    yield dot(g[i*n:(i+1)*n], weight_row)

        def dw(g):
            coords = ((k, j) for j in range(n) for k in range(d)) if tied else (
                (k, j) for k in range(d) for j in range(n))
            for k, j in coords:
                yield total(xrows[i][k] * g[i*n+j] for i in range(rows))

        return self.result((dot(row, col) for row in xrows for col in columns), (a, dx), (w, dw))

    def norm(self, a, gain, d, t, mask, wrong_axis):
        x, weights = self.value(a), self.value(gain)
        if wrong_axis:
            # Run a different normalization: each feature along the sequence.
            out = [0.] * len(x)
            for b in range(len(mask)//t):
                for k in range(d):
                    segment = [(b*t+i)*d+k for i in range(t)]
                    ms = total(x[i]**2 for i in segment) / t
                    r = math.sqrt(1e-5 + ms)
                    for i in segment:
                        if mask[i//d]:
                            out[i] = weights[k]*x[i]/r
            # This mutation is detected by its forward output alone.
            return self.result(out, (a, lambda _: [0.] * len(x)),
                               (gain, lambda _: [0.] * d))
        rms = [math.sqrt(1e-5 + dot(x[i:i+d], x[i:i+d]) / d)
               for i in range(0, len(x), d)]

        def dx(g):
            for row, (valid, r) in enumerate(zip(mask, rms)):
                start = row * d
                xx, gg = x[start:start+d], g[start:start+d]
                product = [u * w for u, w in zip(gg, weights)]
                c = dot(product, xx)
                yield from ((p / r - u*c/(d*r*r*r)) if valid else 0.0
                            for p, u in zip(product, xx))

        def dg(g):
            for k in range(d):
                yield total(g[i*d+k] * x[i*d+k] / rms[i] for i, valid in enumerate(mask) if valid)

        return self.result((weights[i % d]*u/rms[i//d] if mask[i//d] else 0.0
                            for i, u in enumerate(x)), (a, dx), (gain, dg))

    def rope(self, a, t, d, heads, wrong):
        x = self.value(a)
        width = d // heads

        def rotate(values, sign):
            for i in range(0, len(values), 2):
                position = (i // d) % t
                j = (i % width) // 2
                phase = position * 10000 ** (-2 * j / width)
                c, s = f32(math.cos(phase)), sign * f32(math.sin(phase))
                if wrong:
                    s = -s
                u, v = values[i:i+2]
                yield u*c - v*s
                yield u*s + v*c

        return self.result(rotate(x, 1), (a, lambda g: rotate(g, -1)))

    def attention(self, q, k, v, t, d, heads, mask, mutation):
        qv, kv, vv = map(self.value, (q, k, v))
        width = d // heads
        scale = 1.0 if mutation == "scale" else 1 / math.sqrt(width)
        out = [0.] * len(qv)
        rows = []

        def coords(row, head):
            if mutation == "layout":
                return [row*d + j*heads + head for j in range(width)]
            return list(range(row*d + head*width, row*d + (head+1)*width))

        def at(values, indexes):
            return [values[i] for i in indexes]

        for row, valid in enumerate(mask):
            if not valid:
                continue
            start = (row // t) * t
            stop = start + t if mutation == "future" else row + 1
            for head in range(heads):
                a = coords(row, head)
                keys = [coords(j, head) for j in range(start, stop) if mask[j]]
                z = [scale * dot(at(qv, a), at(kv, b)) for b in keys]
                p, _ = softmax(z)
                rows.append((a, keys, p))
                for l, i in enumerate(a):
                    out[i] = total(weight * vv[b[l]] for b, weight in zip(keys, p))

        def derivative(g, side):
            dx = [0.] * len(g)
            for a, keys, probabilities in rows:
                ga = at(g, a)
                dp = [dot(ga, at(vv, b)) for b in keys]
                average = dot(dp, probabilities)
                for b, p, gp in zip(keys, probabilities, dp):
                    ds = p * (gp - average) * scale
                    for i, j in zip(a, b):
                        if side == 0:
                            dx[i] += ds * kv[j]
                        elif side == 1:
                            dx[j] += ds * qv[i]
                        else:
                            dx[j] += p * g[i]
            return dx

        return self.result(out, *((node, lambda g, s=s: derivative(g, s))
                                  for s, node in enumerate((q, k, v))))

    def swiglu(self, a, b, wrong):
        x, y = self.value(a), self.value(b)

        def sigmoid(z):
            e = math.exp(-abs(z))
            return 1/(1+e) if z >= 0 else e/(1+e)

        sig = list(map(sigmoid, x))
        silu = [f32(z*p) for z, p in zip(x, sig)]
        return self.result((u*v for u, v in zip(silu, y)),
                           (a, lambda g: (u*w*(p if wrong else p+z*p*(1-p))
                                          for u, w, p, z in zip(g, y, sig, x))),
                           (b, lambda g: (u*w for u, w in zip(g, silu))))


def dot(a, b):
    # Loop explicitly. Python 3.12 sum(float) uses compensated summation.
    total = 0.
    for x, y in zip(a, b):
        total += x*y
    return total


def total(values):
    result = 0.
    for value in values:
        result += value
    return result


def softmax(z):
    peak = max(z)
    shifted = [math.exp(x-peak) for x in z]
    denominator = total(shifted)
    return [u/denominator for u in shifted], math.log(denominator)


def evaluate(config, params, batch, mutation="", gradients=True, details=None):
    d, n, h, f, v = (config[k] for k in ("d", "n", "h", "f", "v"))
    t = batch["shape"][1]
    ids, targets, mask, loss = (batch[k] for k in ("input", "target", "valid", "loss"))
    graph = Graph()
    leaves = {name: graph.result(values) for name, values in params.items()}
    shared = leaves["embedding"]
    embedding = graph.value(shared)

    def scatter(g):
        out = [0.] * len(embedding)
        for row, (token, valid) in enumerate(zip(ids, mask)):
            if valid:
                for j in range(d):
                    out[token*d+j] += g[row*d+j]
        return out

    def residual(a, b):
        x, y = map(graph.value, (a, b))
        masked = lambda g: (u if mask[i//d] else 0. for i, u in enumerate(g))
        if mutation == "residual":
            return graph.result(masked(y), (b, masked))
        return graph.result(masked(u+w for u, w in zip(x, y)), (a, masked), (b, masked))

    x = graph.result((embedding[token*d+j] if valid else 0.
                      for token, valid in zip(ids, mask) for j in range(d)), (shared, scatter))
    gather = x
    for block in range(n):
        p = {name: leaves[f"block.{block}.{name}"] for name in (
            "attention_gain", "q", "k", "v", "o", "ffn_gain", "gate", "value", "down")}
        u = graph.norm(x, p["attention_gain"], d, t, mask, mutation == "norm")
        q, k, value = [graph.dense(u, p[key], d, d) for key in ("q", "k", "v")]
        qr, kr = [graph.rope(z, t, d, h, mutation == "rope") for z in (q, k)]
        a = graph.attention(qr, kr, value, t, d, h, mask, mutation)
        projected = graph.dense(a, p["o"], d, d)
        xa = residual(x, projected)
        r = graph.norm(xa, p["ffn_gain"], d, t, mask, mutation == "norm")
        g, w = [graph.dense(r, p[key], d, f) for key in ("gate", "value")]
        s = graph.swiglu(g, w, mutation == "derivative")
        down = graph.dense(s, p["down"], f, d)
        x = residual(xa, down)
        if block == 0:
            first = dict(embedding=gather, attention_norm=u, q=q, k=k, v=value,
                         q_rotated=qr, k_rotated=kr, attended=a, projected=projected,
                         attention_residual=xa, ffn_norm=r, gate=g, value=w,
                         activated=s, down=down, block_output=x)
    hidden = graph.norm(x, leaves["final_gain"], d, t, mask, mutation == "norm")
    projection = graph.dense(hidden, shared, d, v, tied=True)
    logits = graph.value(projection)
    rows = []
    for i, (target, valid, supervised) in enumerate(zip(targets, mask, loss)):
        if mutation == "pad" and valid and target == 258:
            supervised = 1
        if not supervised:
            continue
        z = logits[i*v:(i+1)*v]
        if mutation == "shift":
            target = ids[i]
        prob, lse = softmax(z)
        rows.append((i, target, prob, lse + (max(z) - z[target])))
    count = len(rows)
    mean = total(row[-1] for row in rows) / count

    def loss_vjp(g):
        out = [0.] * len(logits)
        for i, target, p, _ in rows:
            out[i*v:(i+1)*v] = [g[0]*(u-(j == target))/(1 if mutation == "mean" else count)
                                for j, u in enumerate(p)]
        return out

    if mutation == "tie":
        graph.nodes[projection] = (logits, graph.nodes[projection][1][:-1])
    root = graph.result([mean], (projection, loss_vjp))
    if not gradients:
        return graph.value(root)[0]
    grads = graph.backward(root)
    if details is not None:
        first.update(final_hidden=hidden, logits=projection, loss=root)
        b = batch["shape"][0]
        for name, node in sorted(first.items()):
            record = graph.value(node)
            shape = [] if name == "loss" else [b, t, len(record)//(b*t)]
            details[name] = dict(shape=shape, value=bits(record), gradient=bits(grads[node]))
    paths = [list(map(f32, scatter(grads[gather]))),
             list(map(f32, graph.nodes[projection][1][-1][1](grads[projection])))
             if mutation != "tie" else [0.] * len(embedding)]
    return logits, graph.value(root), {name: grads[i] for name, i in leaves.items()}, paths


def batch_fixture():
    return dict(shape=[2, 5], input=[3, 7, 3, 11, 258, 10, 12, 9, 258, 258],
                target=[7, 3, 11, 258, 258, 12, 9, 257, 258, 258],
                valid=[1, 1, 1, 1, 0, 1, 1, 1, 0, 0],
                loss=[1, 1, 1, 0, 0, 1, 1, 1, 0, 0])


def finite_differences(config, params, batch, gradients, selected=None):
    results = []
    for step in (0.001, 0.003):
        for name, arr in params.items():
            indexes = [1, len(arr)//2+1]
            if name == "embedding":
                indexes = [1, 3*config["d"] + 1, 7*config["d"] + config["d"]//2]
            if selected is not None:
                indexes = selected.get(name, [])
            for index in indexes:
                old = arr[index]
                arr[index] = f32(old+step)
                high = evaluate(config, params, batch, gradients=False)
                arr[index] = f32(old-step)
                low = evaluate(config, params, batch, gradients=False)
                arr[index] = old
                fd = (high-low)/(2*step)
                analytic = gradients[name][index]
                assert abs(analytic-fd) <= 2e-3 + 2e-2*abs(fd), (name, index, analytic, fd)
                results.append(dict(step=step, name=name, index=index, value=fd))
    return results


def generate():
    config, batch = TINY, batch_fixture()
    params = explicit(config)
    primitive_records = {}
    logits, loss, gradients, paths = evaluate(config, params, batch, details=primitive_records)
    records = {}
    order = list(params)
    for name, shape in inventory(**config):
        records[name] = dict(shape=shape, value=bits(params[name]), gradient=bits(gradients[name]))
    init = {name: bits(values) for name, values in initialized(config)}
    fd = finite_differences(config, params, batch, gradients)
    assert gradients["embedding"] == [f32(a+b) for a, b in zip(*paths)]
    g = Graph()
    q = g.result([1, 0, 0, 1])
    v = g.result([2, 0, 0, 4])
    hand = g.value(g.attention(q, q, v, 2, 2, 1, [1, 1], ""))
    p0 = 1 / (1 + math.exp(1/math.sqrt(2)))
    assert hand == [2.0, 0., f32(2*p0), f32(4*(1-p0))]
    extreme = [f32(10000), f32(9999), f32(-10000)]
    assert abs(softmax(extreme)[1] - math.log1p(math.exp(-1))) < 1e-12
    max_f32 = struct.unpack("<f", bytes.fromhex("ffff7f7f"))[0]
    for z in (extreme, [max_f32, max_f32], [max_f32, f32(max_f32/2)]):
        assert math.isfinite(softmax(z)[1])
        try:
            sum(map(math.exp, z))
        except OverflowError:
            print("naive exponential control: OverflowError")
        else:
            raise AssertionError("negative control survived")
    for mutation in ("scale", "future", "layout", "rope", "norm", "derivative",
                     "residual", "shift", "pad", "tie", "mean"):
        out, wrong_loss, grad, _ = evaluate(config, params, batch, mutation)
        actual = [*out, *wrong_loss]
        wanted = [*logits, *loss]
        if mutation != "norm":
            actual.extend(x for values in grad.values() for x in values)
            wanted.extend(x for values in gradients.values() for x in values)
        delta = max(abs(a-b) for a, b in zip(actual, wanted))
        assert any(abs(a-b) > 2e-5 + 2e-5*abs(b) for a, b in zip(actual, wanted)), mutation
        print(f"{mutation} control: max abs disagreement {delta:.9g}")
    small_params = explicit(SMALL)
    small_logits, small_loss, small_grad, _ = evaluate(SMALL, small_params, batch)
    small_coordinates = {name: [1, len(values)//2+1] for name, values in small_grad.items()}
    small_fd = finite_differences(SMALL, small_params, batch, small_grad,
                                  {"block.0.q": [39], "block.1.o": [17],
                                   "block.2.down": [128], "block.3.ffn_gain": [25]})
    integration = dict(config=SMALL, logits=bits(small_logits), loss=bits(small_loss),
                       finite_differences=small_fd, gradients={name: dict(indexes=indexes, value=bits(small_grad[name][i] for i in indexes))
                                  for name, indexes in small_coordinates.items()})
    return dict(schema_version=1, architecture_version=1, config=config, vocabulary=config["v"],
                seed=f"0x{SEED:016x}", parameter_order=order, batch=batch, parameters=records,
                initialization=init, outputs=dict(logits=bits(logits), loss=bits(loss)),
                embedding_paths=dict(lookup=bits(paths[0]), output=bits(paths[1])),
                finite_differences=fd, hand_attention=bits(hand), integration=integration,
                primitive_records=primitive_records)


def binary32_checksum(values):
    """FNV-1a over the little-endian binary32 byte stream."""
    checksum = 0xCBF29CE484222325
    for value in values:
        for byte in struct.pack("<f", value):
            checksum = ((checksum ^ byte) * 0x100000001B3) & ((1 << 64) - 1)
    return f"{checksum:016x}"


def gradient_summary(values):
    assert all(map(math.isfinite, values))
    return dict(l1=total(map(abs, values)), l2_squared=dot(values, values),
                max_abs=max(map(abs, values)))


def generate_candidate():
    config = CANDIDATE
    params = dict(initialized(config))
    d = config["d"]
    original = list(range(33))
    repeated = original.copy()
    repeated[17] = repeated[3]

    def batch(tokens):
        return dict(shape=[1, 32], input=tokens[:-1], target=tokens[1:],
                    valid=[1]*32, loss=[1]*32)

    batch_original, batch_repeated = batch(original), batch(repeated)
    assert batch_repeated["input"].count(3) == 2
    logits, loss, grad, paths = evaluate(config, params, batch_original)
    _, repeated_loss, repeated_grad, repeated_paths = evaluate(config, params, batch_repeated)
    assert len(logits) == 32*config["v"]
    assert sum(map(len, params.values())) == 914400

    records = []
    for entry, (name, shape) in enumerate(inventory(**config)):
        values = grad[name]
        # Fixed positions in three regions, offset by inventory index.
        region = len(values)//3
        indexes = [entry % region, region + (13*entry) % region,
                   len(values) - 1 - entry % region]
        if name == "embedding":
            # Tokens present in lookup, a token absent from lookup, and the final row.
            indexes = [1, 3*d+17, 17*d+31, 31*d+41, 255*d+53, len(values)-1]
        assert len(indexes) == len(set(indexes))
        records.append(dict(name=name, shape=shape, indexes=indexes,
                            gradient=bits(values[i] for i in indexes),
                            summary=gradient_summary(values),
                            init_checksum=binary32_checksum(params[name])))

    # C2 coordinates are fixed independently of observed gradient errors.
    c2_coordinates = [dict(name=name, index=index,
                           gradient=bits([grad[name][index]])[0])
                      for name, index in C2_COORDINATES]

    def tied_record(gradient, components):
        lookup, output = components
        assert gradient == [f32(a+b) for a, b in zip(lookup, output)]
        # Compare complete E against separate per-path aggregates and selected coordinates.
        indexes = records[0]["indexes"]
        assert all(lookup[i] != 0 and output[i] != 0 for i in indexes[:2])
        assert lookup[indexes[-2]] == 0 and output[indexes[-2]] != 0
        for label, omitted in (("lookup", output), ("output", lookup)):
            error = max(abs(a-b) for a, b in zip(omitted, gradient))
            assert any(abs(a-b) > 2e-5 + 2e-5*abs(b) for a, b in zip(omitted, gradient))
            print(f"Candidate E omission control {label}: max abs disagreement {error:.9g}")
        return dict(indexes=indexes, lookup=bits(lookup[i] for i in indexes),
                    output=bits(output[i] for i in indexes),
                    combined=bits(gradient[i] for i in indexes),
                    summary=gradient_summary(gradient))

    # Use the original batch for FD. Choose positions from the analytic
    # inventory, visiting every block and every family deterministically.
    fd_records = [records[0], *(records[1 + family + 9*(family % config["n"])]
                                for family in range(9)), records[-1]]
    fd = [dict(name=record["name"], index=index)
          for record in fd_records for index in (record["indexes"][:2]
                                                 if record["name"] == "embedding"
                                                 else record["indexes"][1:2])]

    doc = dict(schema_version=1, architecture_version=1, config=config,
                seed=f"0x{SEED:016x}", batch=batch_original,
                logits=bits(logits), loss=bits(loss)[0], parameters=records,
                finite_differences=fd, c2_coordinates=c2_coordinates,
                embedding=tied_record(grad["embedding"], paths),
                repeated=dict(batch=batch_repeated, loss=bits(repeated_loss)[0],
                              embedding=tied_record(repeated_grad["embedding"], repeated_paths)))
    # A distinct explicit-parameter evaluation lets the autograd unit test
    # exercise each native VJP edge before transferring the combined adjoint.
    _, _, explicit_grad, explicit_paths = evaluate(config, explicit(config), batch_repeated)
    indexes = records[0]["indexes"]
    assert all(explicit_paths[0][i] != 0 and explicit_paths[1][i] != 0 for i in indexes[:2])
    path_records = ["# index lookup output combined, IEEE binary32 little-endian hex"]
    for index in indexes:
        entries = [path[index] for path in explicit_paths]
        combined = explicit_grad["embedding"][index]
        assert f32(entries[0] + entries[1]) == combined
        path_records.append(" ".join([str(index), *bits([*entries, combined])]))
    return doc, ("\n".join(path_records) + "\n").encode("utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--candidate", action="store_true", help="verify the full Gate C fixture")
    args = parser.parse_args()
    check_initialization_edge_cases()
    if args.candidate:
        candidate, paths = generate_candidate()
        document("candidate.json", candidate, args.check)
        frozen_payload("candidate-tied-paths.txt", paths, args.check)
        return
    reference = generate()
    parameters = reference.pop("parameters")
    primitives = reference.pop("primitive_records")
    # Keep each canonical document below the repository's artifact-size ceiling.
    documents = {"reference.json": reference,
                 "parameters.json": dict(schema_version=1, parameters=parameters),
                 "operations.json": dict(schema_version=1, records=primitives)}
    for filename, obj in documents.items():
        document(filename, obj, args.check)


def document(filename, obj, check):
    payload = (json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    frozen_payload(filename, payload, check)


def frozen_payload(filename, payload, check):
    path = FIXTURES / filename
    print(f"Transformer {filename}: {len(payload)} bytes, SHA-256 {hashlib.sha256(payload).hexdigest()}")
    if check:
        if payload != path.read_bytes():
            raise SystemExit(f"Frozen Transformer fixture differs: {filename}")
    else:
        path.write_bytes(payload)


if __name__ == "__main__":
    main()
