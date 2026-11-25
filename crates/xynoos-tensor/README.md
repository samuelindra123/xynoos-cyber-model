# XYNOOS tensor foundation

`xynoos-tensor` is a sequential, dependency-free CPU reference library for dense binary32 tensors. The Cargo workspace contains one library crate. The library exposes `Tensor`, `TensorLimits`, and `TensorError`. Internally, `shape.rs` validates shape and offset arithmetic, `tensor.rs` manages ownership and checked access, `ops.rs` implements arithmetic, and `error.rs` defines typed errors. The crate forbids unsafe code.

```rust
use xynoos_tensor::{Tensor, TensorError, TensorLimits};

fn main() -> Result<(), TensorError> {
    let limits = TensorLimits::default();
    let a = Tensor::from_vec(&[2, 3], vec![1., 2., 3., -4., 5., 6.], limits)?;
    let b = Tensor::from_data(&[3, 2], &[7., 8., 9., 10., 11., 12.], limits)?;
    let c = a.matmul(&b, limits)?;
    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.data(), &[58., 64., 83., 90.]);
    assert_eq!(*c.get(&[1, 0])?, 83.);
    Ok(())
}
```

## Representation and indexing

- A tensor privately owns its dimension lengths in a `Vec<usize>` and its values in a contiguous `Vec<f32>`. Rank is the number of dimensions. The last dimension varies fastest. The logical element count equals the data length.
- Rank zero has shape `[]` and contains one element with flat index `0` and multidimensional index `[]`. A positive-rank shape with a zero extent, such as `[0]` or `[2,0,3]`, contains zero elements and every access fails.
- For axis `i`, the canonical stride is the product of dimensions to its right. Construction checks suffix products from right to left. A zero extent makes the element count zero, but an overflowing canonical suffix to its right still gives `ShapeOverflow`. For example `[0, usize::MAX, 2]` is rejected.
- Strides are implicit. Access checks rank and axis bounds, then uses checked row-major Horner arithmetic: each step computes `offset = offset * dim + index`. A flat slice getter checks the final offset. Mutable getters borrow a single value and preserve shape and length.

## API and allocation

In the table, `r` is rank and `n` is the logical element count, except in the matrix product where `n` is the right-hand column count. Complexity includes copying dimension metadata when a tensor is allocated.

| API | Contract | Allocation and time |
|---|---|---|
| `from_vec(shape, data, limits)` | All valid shapes. Requires exactly one input value per element and retained capacity within the byte limit. Consumes the row-major `Vec<f32>` and drops it on error. | Keeps the original numerical buffer on success and copies only the shape, O(r). |
| `from_data(shape, data, limits)` | Same length and layout contract with a borrowed slice. Copies values without arithmetic. | Owns new shape and values, O(r + n). |
| `zeros(shape, limits)`, `filled(shape, value, limits)` | Any valid shape. Uses positive zero or the supplied value, respectively. Accepts all binary32 values. | Owns new shape and values, O(r + n). |
| `shape()`, `rank()`, `numel()`, `data()` | Return the axes, rank, value count and immutable row-major value slice. | Borrow or return by value, O(1). |
| `get_flat(i)`, `get_flat_mut(i)` | Borrow one value at flat index `i < numel`, immutably or mutably. | O(1), no allocation. |
| `get(indices)`, `get_mut(indices)` | Borrow one value at the exact-rank multidimensional index, immutably or mutably. | O(r), no allocation. |
| `add(b, limits)`, `subtract(b, limits)`, `multiply(b, limits)` | Two tensors of identical shape at any rank. Compute binary32 `aᵢ + bᵢ`, `aᵢ − bᵢ`, or `aᵢ × bᵢ`. Preserve shape, including scalar and empty shapes. | Borrow inputs and allocate an independent result, O(r + n). |
| `scale(s, limits)` | Any shape. Computes binary32 `aᵢ × s` for a binary32 scalar. Preserves shape. | Borrow input and allocate an independent result, O(r + n). |
| `dot(b)` | Rank-one vectors of equal length. Multiply binary64 operands and sum products left to right. Returns one `f32`. Empty inputs give positive zero. | Borrow inputs, O(n), no allocation. |
| `sum()` | Any shape. Sums binary64 conversions in increasing flat order. Returns one `f32`. Empty input gives positive zero. | Borrow input, O(n), no allocation or fallible inputs. |
| `matmul(b, limits)` | Rank-two inputs `[m,k]` and `[k,n]` yield `[m,n]`. Each cell is the dot product of the corresponding row and column. A zero output dimension yields an empty result. A zero inner dimension yields positive zeros. | Borrow inputs and allocate one independent output, O(1 + mn + mkn). |
| `transpose(limits)` | Rank-two `[m,n]` becomes `[n,m]` with `out[j,i] = in[i,j]`. Copies values without arithmetic. Empty matrices exchange axes normally. | Borrow input and allocate a contiguous row-major result, O(1 + mn). |

`TensorLimits::default()` applies a 64 MiB ceiling per tensor to both logical bytes (`len * size_of::<f32>()`) and retained numerical capacity (`capacity * size_of::<f32>()`). `TensorLimits::new(bytes)` permits lower limits and clamps larger requests to the ceiling. Either byte count may equal the effective limit. An empty tensor under a zero-byte budget requires zero numerical capacity. Rank is capped at 32, so the requested shape storage is at most `32 * size_of::<usize>()` bytes.

Constructors validate rank, shape products, logical byte count and payload limit, followed by the supplied length when applicable. `from_vec` then checks the caller's existing capacity before copying shape metadata. It preserves accepted spare capacity without copying or shrinking the buffer. Constructors that create a buffer reserve with `try_reserve_exact`, check the capacity reported by `Vec` before writing values, and reject any reservation whose reported capacity exceeds the limit. Allocating operations borrow inputs and apply the supplied limit to their result. Matrix operations check ranks and inner dimensions before validating the output shape and payload.

The budget covers the capacity visible through `Vec::capacity()`. Allocator overhead, allocator rounding beyond reported capacity, shape metadata, other live tensors, aggregate memory and process RSS are outside the budget. A caller-owned buffer already exists when `from_vec` checks it. OS overcommit may still terminate the process after a successful reservation. Matrix work can reach O(mkn) within the memory budget, so callers must bound computation separately.

## Errors

Allocating methods can report `RankLimitExceeded`, `ShapeOverflow`, `ByteCountOverflow`, `PayloadLimitExceeded`, or `AllocationFailed` while validating or reserving. Data constructors also report `DataLengthMismatch`. The capacity check uses checked multiplication, with `RetainedCapacityByteCountOverflow` for a count that cannot be expressed in bytes and `RetainedCapacityLimitExceeded { bytes, limit }` for a representable count above the effective limit. Elementwise operations and vector dot report `ShapeMismatch` for incompatible inputs. Dot, matmul and transpose report `RankMismatch` for the first input of incorrect rank. Matmul reports `MatmulDimensionMismatch` for incompatible inner dimensions. Checked access reports `RankMismatch` or `IndexOutOfBounds` and guards offset arithmetic with `OffsetOverflow`, which is unreachable for an in-bounds index on a valid tensor. Errors implement `Display` and `std::error::Error` and carry relevant sizes or indices without copying the input shape.

## Floating-point behavior

Elementwise arithmetic and scaling operate directly on `f32`. Sum, dot and matmul start each accumulator at positive binary64 zero. They accumulate sequentially and narrow to `f32` once. Dot and matmul widen each operand before multiplication. The product of two finite binary32 values is exactly representable in binary64. Each addition still rounds, so cancellation can discard a small term. For example, summing the binary32 values `[1e30, 1, -1e30]` sequentially returns zero, while grouping the large opposite values first gives one.

NaNs and signed infinities are valid inputs. NaN operands and invalid operations such as opposite infinities added together or zero multiplied by infinity yield NaN. Infinities otherwise follow primitive floating-point rules. Overflow can produce signed infinity. Underflow can produce a subnormal or signed zero. Arithmetic follows the primitive rounding and zero-sign behavior. Positive zero is guaranteed for the specified zero initialization and empty inner products/reductions. Validation compares NaN classification and infinity sign. NaN payloads and bitwise agreement across different implementations or targets are outside the contract.

This reference library implements the operations above. Automatic differentiation, model layers, tokenization, training, inference, quantization, broadcasting, views, SIMD, parallel execution, GPU and C integration remain deferred.
