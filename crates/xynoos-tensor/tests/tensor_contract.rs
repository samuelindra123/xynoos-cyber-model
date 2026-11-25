#![forbid(unsafe_code)]

use xynoos_tensor::{Tensor, TensorError, TensorLimits};

const LIMITS: TensorLimits = TensorLimits::new(1024);

fn tensor(shape: &[usize], data: &[f32]) -> Tensor {
    Tensor::from_data(shape, data, LIMITS).unwrap()
}

#[test]
fn ownership_and_row_major_access() {
    let values = vec![0., 1., 2., 3., 4., 5., 6., 7.];
    let pointer = values.as_ptr();
    let mut owned = Tensor::from_vec(&[2, 2, 2], values, LIMITS).unwrap();
    assert_eq!(owned.data().as_ptr(), pointer);
    assert_eq!(owned.shape(), [2, 2, 2]);
    assert_eq!(owned.rank(), 3);
    assert_eq!(owned.numel(), 8);
    assert_eq!(owned.get(&[0, 0, 0]), Ok(&0.));
    assert_eq!(owned.get(&[1, 0, 1]), Ok(&5.));
    assert_eq!(owned.get(&[1, 1, 1]), Ok(&7.));
    assert_eq!(owned.get_flat(5), Ok(&5.));
    *owned.get_mut(&[1, 0, 1]).unwrap() = -5.;
    *owned.get_flat_mut(7).unwrap() = 20.;
    assert_eq!(owned.data(), [0., 1., 2., 3., 4., -5., 6., 20.]);

    let original = [1., -0.0];
    let mut copy = tensor(&[2], &original);
    assert_eq!(copy.data()[1].to_bits(), (-0.0f32).to_bits());
    *copy.get_flat_mut(0).unwrap() = 9.;
    assert_eq!(original[0], 1.);
    assert_eq!(
        Tensor::from_data(&[1], &[], LIMITS).unwrap_err(),
        TensorError::DataLengthMismatch {
            expected: 1,
            actual: 0
        }
    );
    assert_eq!(
        Tensor::from_vec(&[0], vec![1.], LIMITS).unwrap_err(),
        TensorError::DataLengthMismatch {
            expected: 0,
            actual: 1
        }
    );
}

#[test]
fn constructors_distinguish_scalars_and_empties() {
    let mut scalar = Tensor::filled(&[], -4., LIMITS).unwrap();
    assert_eq!(scalar.rank(), 0);
    assert_eq!(scalar.numel(), 1);
    assert_eq!(scalar.get(&[]), Ok(&-4.));
    *scalar.get_mut(&[]).unwrap() = 2.;
    assert_eq!(scalar.data(), [2.]);
    assert_eq!(scalar.sum(), 2.);
    assert_eq!(
        scalar.get(&[0]),
        Err(TensorError::RankMismatch {
            expected: 0,
            actual: 1
        })
    );

    let empty = Tensor::zeros(&[2, 0, 3], TensorLimits::new(0)).unwrap();
    assert_eq!(empty.rank(), 3);
    assert_eq!(empty.numel(), 0);
    assert!(empty.data().is_empty());
    assert_eq!(empty.sum().to_bits(), 0);
    assert_eq!(
        empty.get(&[0, 0, 2]),
        Err(TensorError::IndexOutOfBounds {
            axis: Some(1),
            index: 0,
            len: 0
        })
    );
    assert_eq!(
        empty.get_flat(0),
        Err(TensorError::IndexOutOfBounds {
            axis: None,
            index: 0,
            len: 0
        })
    );

    let zeros = Tensor::zeros(&[2], LIMITS).unwrap();
    assert!(zeros.data().iter().all(|x| x.to_bits() == 0));
    let constants = Tensor::filled(&[2, 2], -3., LIMITS).unwrap();
    assert_eq!(constants.data(), [-3.; 4]);
    assert!(Tensor::filled(&[], f32::NAN, LIMITS)
        .unwrap()
        .sum()
        .is_nan());
    assert_eq!(
        Tensor::filled(&[], f32::INFINITY, LIMITS).unwrap().sum(),
        f32::INFINITY
    );
}

#[test]
fn indexing_checks_rank_and_each_axis() {
    let mut data = Tensor::zeros(&[2, 3, 4], LIMITS).unwrap();
    assert_eq!(
        data.get(&[0, 0]),
        Err(TensorError::RankMismatch {
            expected: 3,
            actual: 2
        })
    );
    assert_eq!(
        data.get_mut(&[0, 0, 0, 0]),
        Err(TensorError::RankMismatch {
            expected: 3,
            actual: 4
        })
    );
    for (axis, len) in [2, 3, 4].into_iter().enumerate() {
        let mut indices = [0; 3];
        indices[axis] = len;
        assert_eq!(
            data.get_mut(&indices),
            Err(TensorError::IndexOutOfBounds {
                axis: Some(axis),
                index: len,
                len
            })
        );
    }
    assert_eq!(
        data.get_flat_mut(24),
        Err(TensorError::IndexOutOfBounds {
            axis: None,
            index: 24,
            len: 24
        })
    );

    let wide_empty = tensor(&[usize::MAX, usize::MAX, 0], &[]);
    assert_eq!(
        wide_empty.get(&[2, 0, 0]),
        Err(TensorError::IndexOutOfBounds {
            axis: Some(2),
            index: 0,
            len: 0
        })
    );
}

#[test]
fn rejects_shape_byte_and_limit_violations_before_reservation() {
    assert_eq!(
        Tensor::zeros(&[usize::MAX, 2], LIMITS).unwrap_err(),
        TensorError::ShapeOverflow { axis: 0 }
    );
    assert_eq!(
        Tensor::zeros(&[0, usize::MAX, 2], LIMITS).unwrap_err(),
        TensorError::ShapeOverflow { axis: 1 }
    );
    assert_eq!(
        Tensor::zeros(&[usize::MAX], LIMITS).unwrap_err(),
        TensorError::ByteCountOverflow {
            elements: usize::MAX
        }
    );

    let too_many_axes = [1; TensorLimits::MAX_RANK + 1];
    assert_eq!(
        Tensor::zeros(&too_many_axes, LIMITS).unwrap_err(),
        TensorError::RankLimitExceeded {
            rank: too_many_axes.len(),
            limit: TensorLimits::MAX_RANK
        }
    );
    assert_eq!(
        Tensor::zeros(&[1; TensorLimits::MAX_RANK], LIMITS)
            .unwrap()
            .numel(),
        1
    );
    let tiny = TensorLimits::new(4);
    assert_eq!(
        TensorLimits::new(usize::MAX).max_bytes(),
        TensorLimits::MAX_BYTES
    );
    match Tensor::zeros(&[1], tiny) {
        Ok(tensor) => assert_eq!(tensor.data(), [0.]),
        Err(TensorError::RetainedCapacityLimitExceeded { bytes, limit }) => {
            assert_eq!(limit, tiny.max_bytes());
            assert!(bytes > limit);
        }
        Err(error) => panic!("unexpected error at logical payload boundary: {error}"),
    }
    let expected = TensorError::PayloadLimitExceeded { bytes: 8, limit: 4 };
    assert_eq!(Tensor::zeros(&[2], tiny).unwrap_err(), expected);
    assert_eq!(
        Tensor::from_vec(&[2], vec![1., 2.], tiny).unwrap_err(),
        expected
    );
    assert_eq!(
        Tensor::from_data(&[2], &[1., 2.], tiny).unwrap_err(),
        expected
    );
}

#[test]
fn elementwise_operations_and_limits() {
    let left = tensor(&[1, 3], &[1., -2., 0.]);
    let mut right = tensor(&[1, 3], &[3., 4., -2.]);
    let added = left.add(&right, LIMITS).unwrap();
    assert_eq!(added.shape(), [1, 3]);
    assert_eq!(added.data(), [4., 2., -2.]);
    assert_eq!(
        left.subtract(&right, LIMITS).unwrap().data(),
        [-2., -6., 2.]
    );
    assert_eq!(left.multiply(&right, LIMITS).unwrap().data(), [3., -8., 0.]);
    assert_eq!(left.scale(-2., LIMITS).unwrap().data(), [-2., 4., 0.]);

    *right.get_flat_mut(0).unwrap() = 20.;
    assert_eq!(added.data(), [4., 2., -2.]);
    let mismatch = tensor(&[3, 1], &[1., 2., 3.]);
    let expected = TensorError::ShapeMismatch {
        axis: Some(0),
        left: 1,
        right: 3,
    };
    assert_eq!(left.add(&mismatch, LIMITS).unwrap_err(), expected);
    assert_eq!(left.subtract(&mismatch, LIMITS).unwrap_err(), expected);
    assert_eq!(left.multiply(&mismatch, LIMITS).unwrap_err(), expected);
    assert_eq!(
        left.add(&tensor(&[3], &[1.; 3]), LIMITS).unwrap_err(),
        TensorError::ShapeMismatch {
            axis: None,
            left: 2,
            right: 1
        }
    );

    let empty = tensor(&[0], &[]);
    let zero_budget = TensorLimits::new(0);
    assert_eq!(empty.add(&empty, zero_budget).unwrap().shape(), [0]);
    assert_eq!(empty.subtract(&empty, zero_budget).unwrap().shape(), [0]);
    assert_eq!(empty.multiply(&empty, zero_budget).unwrap().shape(), [0]);
    assert_eq!(empty.scale(f32::NAN, zero_budget).unwrap().shape(), [0]);
    let small = TensorLimits::new(11);
    let expected = TensorError::PayloadLimitExceeded {
        bytes: 12,
        limit: 11,
    };
    assert_eq!(left.add(&right, small).unwrap_err(), expected);
    assert_eq!(left.subtract(&right, small).unwrap_err(), expected);
    assert_eq!(left.multiply(&right, small).unwrap_err(), expected);
    assert_eq!(left.scale(1., small).unwrap_err(), expected);
}

#[test]
fn reductions_obey_order_precision_and_empty_identity() {
    let left = tensor(&[3], &[-2., 1., 3.]);
    let right = tensor(&[3], &[3., 2., -4.]);
    assert_eq!(left.sum(), 2.);
    assert_eq!(left.dot(&right), Ok(-16.));
    assert_eq!(tensor(&[1], &[3.]).dot(&tensor(&[1], &[-2.])), Ok(-6.));
    let empty = tensor(&[0], &[]);
    assert_eq!(empty.dot(&empty).unwrap().to_bits(), 0);
    assert_eq!(
        empty.dot(&right),
        Err(TensorError::ShapeMismatch {
            axis: Some(0),
            left: 0,
            right: 3
        })
    );
    let scalar = tensor(&[], &[1.]);
    assert_eq!(
        scalar.dot(&right),
        Err(TensorError::RankMismatch {
            expected: 1,
            actual: 0
        })
    );
    assert_eq!(
        right.dot(&scalar),
        Err(TensorError::RankMismatch {
            expected: 1,
            actual: 0
        })
    );

    // The unit term survives cancellation at this scale in f64, but not f32.
    let values = tensor(&[3], &[16_777_216., 1., -16_777_216.]);
    assert_eq!(values.sum(), 1.);
    assert_eq!(values.dot(&tensor(&[3], &[1.; 3])), Ok(1.));
    // A unit term can still be lost by binary64 when the exponent gap is large.
    let ordered = tensor(&[3], &[1.0e30, 1., -1.0e30]);
    let reordered = tensor(&[3], &[1.0e30, -1.0e30, 1.]);
    let ones = tensor(&[3], &[1.; 3]);
    assert_eq!(ordered.sum().to_bits(), 0);
    assert_eq!(ordered.dot(&ones).unwrap().to_bits(), 0);
    assert_eq!(reordered.sum(), 1.);
    assert_eq!(reordered.dot(&ones), Ok(1.));
    assert_eq!(
        tensor(&[1, 3], ordered.data())
            .matmul(&tensor(&[3, 1], ones.data()), LIMITS)
            .unwrap()
            .data()[0]
            .to_bits(),
        0
    );
}

#[test]
fn floating_point_classification_and_conversion() {
    let max = f32::MAX;
    let min = f32::from_bits(1);
    let vector = tensor(&[2], &[max, -max]);
    assert_eq!(
        vector.scale(2., LIMITS).unwrap().data(),
        [f32::INFINITY, f32::NEG_INFINITY]
    );
    assert_eq!(vector.dot(&tensor(&[2], &[max; 2])).unwrap().to_bits(), 0);
    assert_eq!(tensor(&[2], &[max; 2]).sum(), f32::INFINITY);
    assert_eq!(tensor(&[2], &[-max; 2]).sum(), f32::NEG_INFINITY);
    assert_eq!(tensor(&[1], &[min]).sum().to_bits(), 1);
    assert_eq!(
        tensor(&[1], &[min]).scale(0.5, LIMITS).unwrap().data()[0].to_bits(),
        0
    );
    assert_eq!(
        tensor(&[1], &[-min])
            .dot(&tensor(&[1], &[0.5]))
            .unwrap()
            .to_bits(),
        (-0.0f32).to_bits()
    );

    let infinite = tensor(&[2], &[f32::INFINITY, f32::NEG_INFINITY]);
    assert!(infinite.sum().is_nan());
    assert!(infinite.dot(&tensor(&[2], &[1.; 2])).unwrap().is_nan());
    assert!(infinite
        .scale(0., LIMITS)
        .unwrap()
        .data()
        .iter()
        .all(|x| x.is_nan()));
    assert!(tensor(&[1, 1], &[f32::NEG_INFINITY])
        .matmul(&tensor(&[1, 1], &[0.]), LIMITS)
        .unwrap()
        .data()[0]
        .is_nan());
}

#[test]
fn matrix_product_layout_empty_cases_and_errors() {
    let left = tensor(&[2, 3], &[1., 2., 3., -4., 5., 6.]);
    let right = tensor(&[3, 2], &[7., 8., 9., 10., 11., 12.]);
    let product = left.matmul(&right, LIMITS).unwrap();
    assert_eq!(product.shape(), [2, 2]);
    assert_eq!(product.data(), [58., 64., 83., 90.]);

    let square = tensor(&[2, 2], &[2., 3., -4., 7.]);
    assert_eq!(
        square.matmul(&square, LIMITS).unwrap().data(),
        [-8., 27., -36., 37.]
    );
    let identity = tensor(&[2, 2], &[1., 0., 0., 1.]);
    assert_eq!(
        square.matmul(&identity, LIMITS).unwrap().data(),
        square.data()
    );
    assert_eq!(
        identity.matmul(&square, LIMITS).unwrap().data(),
        square.data()
    );
    assert_eq!(
        square
            .matmul(&Tensor::zeros(&[2, 3], LIMITS).unwrap(), LIMITS)
            .unwrap()
            .data(),
        [0.; 6]
    );

    let zero_inner = tensor(&[2, 0], &[])
        .matmul(&tensor(&[0, 3], &[]), LIMITS)
        .unwrap();
    assert_eq!(zero_inner.shape(), [2, 3]);
    assert!(zero_inner.data().iter().all(|x| x.to_bits() == 0));
    for (lhs, rhs, expected) in [
        ([0, 3], [3, 2], [0, 2]),
        ([2, 3], [3, 0], [2, 0]),
        ([0, 0], [0, 0], [0, 0]),
        ([usize::MAX, 0], [0, 0], [usize::MAX, 0]),
    ] {
        let lhs = Tensor::zeros(&lhs, LIMITS).unwrap();
        let rhs = Tensor::zeros(&rhs, LIMITS).unwrap();
        let empty = lhs.matmul(&rhs, TensorLimits::new(0)).unwrap();
        assert_eq!(empty.shape(), expected);
        assert_eq!(empty.numel(), 0);
    }

    assert_eq!(
        left.matmul(&left, LIMITS).unwrap_err(),
        TensorError::MatmulDimensionMismatch { left: 3, right: 2 }
    );
    let vector = tensor(&[2], &[1.; 2]);
    for error in [left.matmul(&vector, LIMITS), vector.matmul(&right, LIMITS)] {
        assert_eq!(
            error.unwrap_err(),
            TensorError::RankMismatch {
                expected: 2,
                actual: 1
            }
        );
    }
    assert_eq!(
        left.matmul(&right, TensorLimits::new(15)).unwrap_err(),
        TensorError::PayloadLimitExceeded {
            bytes: 16,
            limit: 15
        }
    );

    let wide = tensor(&[usize::MAX, 0], &[]);
    assert_eq!(
        wide.matmul(&tensor(&[0, 2], &[]), LIMITS).unwrap_err(),
        TensorError::ShapeOverflow { axis: 0 }
    );
    assert_eq!(
        wide.matmul(&tensor(&[0, 1], &[]), LIMITS).unwrap_err(),
        TensorError::ByteCountOverflow {
            elements: usize::MAX
        }
    );
}

#[test]
fn transpose_makes_owned_row_major_copy() {
    let values = [1., -2., 3., 4., 5., -6.];
    let input = tensor(&[2, 3], &values);
    let mut transposed = input.transpose(LIMITS).unwrap();
    assert_eq!(transposed.shape(), [3, 2]);
    assert_eq!(transposed.data(), [1., 4., -2., 5., 3., -6.]);
    let roundtrip = transposed.transpose(LIMITS).unwrap();
    assert_eq!(roundtrip.shape(), input.shape());
    assert_eq!(roundtrip.data(), values);
    *transposed.get_mut(&[0, 0]).unwrap() = 16.;
    assert_eq!(input.data(), values);
    for (shape, output_shape) in [([1, 3], [3, 1]), ([3, 1], [1, 3])] {
        let vector_matrix = tensor(&shape, &values[..3]);
        let output = vector_matrix.transpose(LIMITS).unwrap();
        assert_eq!(output.shape(), output_shape);
        assert_eq!(output.data(), &values[..3]);
    }
    let specials = tensor(&[2, 2], &[f32::NAN, -0.0, f32::NEG_INFINITY, 3.]);
    let output = specials.transpose(LIMITS).unwrap();
    assert!(output.data()[0].is_nan());
    assert_eq!(output.data()[1], f32::NEG_INFINITY);
    assert_eq!(output.data()[2].to_bits(), (-0.0f32).to_bits());
    assert_eq!(output.data()[3], 3.);
    for (shape, expected) in [([0, 2], [2, 0]), ([3, 0], [0, 3])] {
        let output = tensor(&shape, &[]).transpose(TensorLimits::new(0)).unwrap();
        assert_eq!(output.shape(), expected);
        assert_eq!(output.numel(), 0);
    }
    assert_eq!(
        tensor(&[], &[1.]).transpose(LIMITS).unwrap_err(),
        TensorError::RankMismatch {
            expected: 2,
            actual: 0
        }
    );
    assert_eq!(
        input.transpose(TensorLimits::new(23)).unwrap_err(),
        TensorError::PayloadLimitExceeded {
            bytes: 24,
            limit: 23
        }
    );
}

#[test]
fn bounded_exact_properties_for_small_integers() {
    for rows in 0..=4 {
        for columns in 0..=4 {
            let data: Vec<_> = (0..rows * columns).map(|i| (i % 7) as f32 - 3.).collect();
            let a = tensor(&[rows, columns], &data);
            let zero = Tensor::zeros(a.shape(), LIMITS).unwrap();
            let ones = Tensor::filled(a.shape(), 1., LIMITS).unwrap();
            let added = a.add(&zero, LIMITS).unwrap();
            assert_eq!(added.shape(), a.shape());
            assert_eq!(added.data(), data);
            let subtracted = a.subtract(&zero, LIMITS).unwrap();
            assert_eq!(subtracted.shape(), a.shape());
            assert_eq!(subtracted.data(), data);
            assert_eq!(a.multiply(&ones, LIMITS).unwrap().data(), data);
            let restored = a.transpose(LIMITS).unwrap().transpose(LIMITS).unwrap();
            assert_eq!(restored.shape(), a.shape());
            assert_eq!(restored.data(), data);

            let split = data.len() / 2;
            let lhs = tensor(&[split], &data[..split]);
            let rhs = tensor(&[data.len() - split], &data[split..]);
            assert_eq!(a.sum(), lhs.sum() + rhs.sum());
            let reversed: Vec<_> = data.iter().copied().rev().collect();
            let x = tensor(&[data.len()], &data);
            let y = tensor(&[data.len()], &reversed);
            assert_eq!(x.dot(&y), y.dot(&x));
        }
    }
}
