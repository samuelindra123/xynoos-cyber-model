"""Independent binary64 AdamW and token-weighted clipping reference."""

import argparse
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import struct

FIXTURE = Path(__file__).resolve().parent / "fixtures/task006-adamw.tsv"
U64_MAX = (1 << 64) - 1
F32_MAX = struct.unpack("<f", bytes.fromhex("ffff7f7f"))[0]


class Rejected(Exception):
    pass


def f32(value):
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def bits(value):
    return f"{struct.unpack('<I', struct.pack('<f', f32(value)))[0]:08x}"


def bits64(value):
    return f"{struct.unpack('<Q', struct.pack('<d', value))[0]:016x}"


def finite(value):
    return math.isfinite(value)


@dataclass(frozen=True)
class Config:
    beta1: float
    beta2: float
    epsilon: float
    weight_decay: float

    @classmethod
    def make(cls, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.1):
        return cls(*(f32(value) for value in (beta1, beta2, epsilon, weight_decay)))


@dataclass(frozen=True)
class State:
    parameters: tuple
    first: tuple
    second: tuple
    beta1_power: float
    beta2_power: float
    step: int


def validate(config, learning_rate):
    values = (config.beta1, config.beta2, config.epsilon, config.weight_decay, learning_rate)
    if not all(finite(value) for value in values):
        raise Rejected("invalid_hyperparameter")
    if not 0 <= config.beta1 < 1 or not 0 <= config.beta2 < 1:
        raise Rejected("invalid_hyperparameter")
    if config.epsilon <= 0 or config.weight_decay < 0 or learning_rate <= 0:
        raise Rejected("invalid_hyperparameter")
    if learning_rate * config.weight_decay >= 1:
        raise Rejected("invalid_decay_product")


def step(state, gradients, config, learning_rate):
    """Stage a complete scalar/vector update; return a new immutable state."""
    validate(config, learning_rate)
    if state.step < 0 or state.step > U64_MAX:
        raise Rejected("invalid_step")
    if state.step == U64_MAX:
        raise Rejected("step_overflow")
    if not (
        len(state.parameters) == len(state.first) == len(state.second) == len(gradients)
    ):
        raise Rejected("shape_mismatch")
    if not all(finite(value) for values in
               (state.parameters, state.first, state.second, gradients)
               for value in values):
        raise Rejected("non_finite_input")
    if not finite(state.beta1_power) or not finite(state.beta2_power):
        raise Rejected("non_finite_state")
    if not (0 <= state.beta1_power <= 1 and 0 <= state.beta2_power <= 1):
        raise Rejected("invalid_bias_power")
    if any(value < 0 for value in state.second):
        raise Rejected("invalid_second_moment")

    beta1 = config.beta1
    beta2 = config.beta2
    epsilon = config.epsilon
    decay = config.weight_decay
    next_step = state.step + 1
    next_beta1_power = state.beta1_power * beta1
    next_beta2_power = state.beta2_power * beta2
    correction1 = 1.0 - next_beta1_power
    correction2 = 1.0 - next_beta2_power
    if not (finite(correction1) and finite(correction2) and correction1 > 0 and correction2 > 0):
        raise Rejected("invalid_bias_correction")

    decay_scale = 1.0 - learning_rate * decay
    next_parameters = []
    next_first = []
    next_second = []
    for parameter, old_first, old_second, gradient in zip(
        state.parameters, state.first, state.second, gradients
    ):
        first_left = beta1 * old_first
        first_right = (1.0 - beta1) * gradient
        first = f32(first_left + first_right)

        gradient_square = gradient * gradient
        second_left = beta2 * old_second
        second_right = (1.0 - beta2) * gradient_square
        second = f32(second_left + second_right)
        if not (finite(first) and finite(second)) or second < 0:
            raise Rejected("moment_overflow")

        corrected_first = first / correction1
        corrected_second = second / correction2
        if not (finite(corrected_first) and finite(corrected_second)) or corrected_second < 0:
            raise Rejected("corrected_moment_non_finite")
        denominator = math.sqrt(corrected_second) + epsilon
        if not finite(denominator) or denominator <= 0:
            raise Rejected("invalid_denominator")
        decayed = decay_scale * parameter
        adaptive = learning_rate * corrected_first / denominator
        candidate = f32(decayed - adaptive)
        if not finite(candidate):
            raise Rejected("parameter_overflow")

        next_parameters.append(candidate)
        next_first.append(first)
        next_second.append(second)

    if not (finite(next_beta1_power) and finite(next_beta2_power)):
        raise Rejected("bias_power_non_finite")
    return State(tuple(next_parameters), tuple(next_first), tuple(next_second),
                 next_beta1_power, next_beta2_power, next_step)


def fresh(parameters):
    values = tuple(f32(value) for value in parameters)
    if not all(finite(value) for value in values):
        raise Rejected("non_finite_parameter")
    return State(values, (0.0,) * len(values), (0.0,) * len(values), 1.0, 1.0, 0)


def weighted_mean(gradient_means, target_counts):
    if len(gradient_means) != len(target_counts):
        raise Rejected("shape_mismatch")
    if any(not isinstance(count, int) or isinstance(count, bool) or count < 0 or count > U64_MAX
           for count in target_counts):
        raise Rejected("invalid_target_count")
    total = sum(target_counts)
    if total > U64_MAX:
        raise Rejected("target_count_overflow")
    if total == 0:
        raise Rejected("empty_supervision")
    if not gradient_means:
        raise Rejected("shape_mismatch")
    width = len(gradient_means[0])
    if any(len(row) != width for row in gradient_means):
        raise Rejected("shape_mismatch")
    sums = [0.0] * width
    for row, count in zip(gradient_means, target_counts):
        for index, value in enumerate(row):
            if not finite(value):
                raise Rejected("non_finite_input")
            product = float(value) * count
            sums[index] = sums[index] + product
    return tuple(f32(total_value / total) for total_value in sums)


def global_clip(gradient, threshold):
    if not finite(threshold) or threshold <= 0:
        raise Rejected("invalid_clip_threshold")
    if not all(finite(value) for value in gradient):
        raise Rejected("non_finite_input")
    total = 0.0
    for value in gradient:
        square = float(value) * float(value)
        total = total + square
    norm = math.sqrt(total)
    if not finite(norm):
        raise Rejected("non_finite_norm")
    if norm == 0 or norm <= threshold:
        return tuple(gradient), norm, False
    scale = threshold / norm
    return tuple(f32(float(value) * scale) for value in gradient), norm, True


def row(name, *values):
    return name + "\t" + "\t".join(str(value) for value in values) + "\n"


def snapshot_row(name, state):
    values = [str(state.step), bits64(state.beta1_power), bits64(state.beta2_power)]
    for parameter, first, second in zip(state.parameters, state.first, state.second):
        values.extend((bits(parameter), bits(first), bits(second)))
    return row(name, *values)


def require_rejection(state, gradients, config, rate, expected):
    before = state
    try:
        step(state, gradients, config, rate)
    except Rejected as error:
        if str(error) != expected:
            raise AssertionError(f"expected {expected}, received {error}")
    else:
        raise AssertionError(f"expected rejection {expected}")
    if state != before:
        raise AssertionError("rejected update mutated its input state")


def fixture_rows():
    output = ["# XYNOOS TASK 006 AdamW independent reference v1\n"]
    default = Config.make()

    first = step(fresh([1.0]), [2.0], default, f32(0.01))
    output.append(snapshot_row("first_step", first))

    multi = fresh([1.25, -0.5])
    for index, gradient in enumerate(((2.0, -1.0), (-1.0, 0.5), (0.0, 0.0), (0.5, -2.0)), 1):
        multi = step(multi, gradient, default, f32(0.01))
        output.append(snapshot_row(f"multi_{index}", multi))

    zero = step(fresh([2.0]), [0.0], default, f32(0.1))
    output.append(snapshot_row("zero_gradient_with_decay", zero))
    if zero.first != (0.0,) or zero.second != (0.0,) or zero.parameters != (f32(1.98),):
        raise AssertionError("zero-gradient decoupled decay behavior changed")

    epsilon_config = Config.make(beta1=0.0, beta2=0.0, epsilon=0.1, weight_decay=0.0)
    epsilon_state = step(fresh([0.0]), [0.02], epsilon_config, f32(0.01))
    inside_sqrt = f32(-0.01 * 0.02 / math.sqrt(0.02 * 0.02 + epsilon_config.epsilon))
    output.append(row("epsilon_outside_sqrt", bits(epsilon_state.parameters[0])))
    output.append(row("negative_control_epsilon_inside_sqrt", bits(inside_sqrt)))
    if epsilon_state.parameters[0] == inside_sqrt:
        raise AssertionError("epsilon placement negative control did not distinguish")

    decay_config = Config.make(beta1=0.0, beta2=0.0, epsilon=0.1, weight_decay=0.5)
    decoupled = step(fresh([1.0]), [1.0], decay_config, f32(0.1))
    coupled_gradient = f32(1.0 + decay_config.weight_decay * 1.0)
    coupled = f32(1.0 - f32(0.1) * coupled_gradient /
                  (abs(coupled_gradient) + decay_config.epsilon))
    output.append(row("decoupled_weight_decay", bits(decoupled.parameters[0])))
    output.append(row("negative_control_coupled_l2", bits(coupled)))
    if decoupled.parameters[0] == coupled:
        raise AssertionError("coupled L2 negative control did not distinguish")

    bias_config = Config.make(beta1=0.5, beta2=0.5, epsilon=1e-6, weight_decay=0.0)
    corrected = step(fresh([0.5]), [2.0], bias_config, f32(0.1))
    uncorrected = f32(0.5 - f32(0.1) * corrected.first[0] /
                      (math.sqrt(corrected.second[0]) + bias_config.epsilon))
    output.append(row("bias_corrected", bits(corrected.parameters[0]),
                      bits(corrected.first[0]), bits(corrected.second[0])))
    output.append(row("negative_control_no_bias_correction", bits(uncorrected)))
    if corrected.parameters[0] == uncorrected:
        raise AssertionError("bias-correction negative control did not distinguish")

    means = ((2.0, 0.0), (0.0, 3.0), (9.0, 9.0))
    token_weighted = weighted_mean(means, (2, 1, 0))
    naive = tuple(f32((means[0][i] + means[1][i]) / 2.0) for i in range(2))
    clipped, norm, changed = global_clip(token_weighted, 1.0)
    wrong_clipped, wrong_norm, wrong_changed = global_clip(naive, 1.0)
    equal_boundary, _, equal_changed = global_clip(token_weighted, norm)
    above_boundary, _, above_changed = global_clip(token_weighted, 2.0)
    output.append(row("token_weighted_mean", *(bits(value) for value in token_weighted)))
    output.append(row("negative_control_equal_microbatch_means", *(bits(value) for value in naive)))
    output.append(row("global_clip_below_threshold", bits(norm), *(bits(value) for value in clipped), str(changed)))
    output.append(row("negative_control_clip_wrong_mean", bits(wrong_norm),
                      *(bits(value) for value in wrong_clipped), str(wrong_changed)))
    output.append(row("global_clip_at_threshold", *(bits(value) for value in equal_boundary),
                      str(equal_changed)))
    output.append(row("global_clip_above_threshold", *(bits(value) for value in above_boundary),
                      str(above_changed)))
    if token_weighted != (f32(4.0 / 3.0), 1.0) or naive != (1.0, 1.5):
        raise AssertionError("token-weighted accumulation example changed")
    if not changed or wrong_clipped == clipped or equal_changed or above_changed:
        raise AssertionError("gradient clipping controls did not distinguish")
    try:
        weighted_mean(((1.0,),), (0,))
    except Rejected as error:
        if str(error) != "empty_supervision":
            raise
    else:
        raise AssertionError("empty supervision must not produce an update")
    output.append(row("reject_empty_supervision", "empty_supervision"))
    extreme, extreme_norm, extreme_changed = global_clip((F32_MAX, -F32_MAX), 1.0)
    if not extreme_changed or not finite(extreme_norm) or not all(finite(value) for value in extreme):
        raise AssertionError("extreme finite gradients should clip to finite values")
    output.append(row("extreme_finite_gradient_clip", bits64(extreme_norm),
                      *(bits(value) for value in extreme), str(extreme_changed)))

    invalid_cases = [
        ("zero_learning_rate", default, 0.0, "invalid_hyperparameter"),
        ("beta1_one", Config.make(beta1=1.0), 0.01, "invalid_hyperparameter"),
        ("negative_beta2", Config.make(beta2=-0.1), 0.01, "invalid_hyperparameter"),
        ("zero_epsilon", Config.make(epsilon=0.0), 0.01, "invalid_hyperparameter"),
        ("negative_decay", Config.make(weight_decay=-1.0), 0.01, "invalid_hyperparameter"),
        ("non_finite_beta", Config.make(beta1=math.nan), 0.01, "invalid_hyperparameter"),
        ("decay_product_at_one", Config.make(weight_decay=10.0), 0.1, "invalid_decay_product"),
    ]
    for name, config, rate, reason in invalid_cases:
        require_rejection(fresh([1.0]), [1.0], config, f32(rate), reason)
        output.append(row("reject_" + name, reason))

    nonfinite_gradient = fresh([1.0])
    require_rejection(nonfinite_gradient, [math.nan], default, f32(0.01), "non_finite_input")
    output.append(row("reject_non_finite_gradient", "non_finite_input"))

    overflow_state = fresh([1.0, 1.0])
    overflow_config = Config.make(beta1=0.0, beta2=0.0, epsilon=1e-8, weight_decay=0.0)
    require_rejection(overflow_state, [1.0, F32_MAX], overflow_config, f32(0.01), "moment_overflow")
    output.append(row("reject_f32_moment_overflow", "moment_overflow"))

    boundary = State((1.0,), (0.0,), (0.0,), 1.0, 1.0, U64_MAX - 1)
    last = step(boundary, [0.0], Config.make(weight_decay=0.0), f32(0.01))
    output.append(row("step_counter_last_value", str(last.step)))
    require_rejection(last, [0.0], Config.make(weight_decay=0.0), f32(0.01), "step_overflow")
    output.append(row("reject_step_counter_overflow", str(U64_MAX)))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args()
    data = "".join(fixture_rows()).encode("ascii")
    if args.write:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_bytes(data)
    elif FIXTURE.read_bytes() != data:
        raise SystemExit(f"{FIXTURE.relative_to(FIXTURE.parents[1])}: reference fixture differs")
    print(f"{FIXTURE.relative_to(FIXTURE.parents[1])}: {len(data.splitlines()) - 1} records, "
          f"SHA-256 {hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
