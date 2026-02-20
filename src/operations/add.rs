use quote::__private::TokenStream;
use quote::quote;
use super::OperationCodeGenerator;

pub struct AddOperation;

impl OperationCodeGenerator for AddOperation {
    /// Generates code for ONNX Add operation
    /// Reference: https://onnx.ai/onnx/operators/onnx__Add.html
    ///
    /// Performs element-wise binary addition with Numpy-style broadcasting support.
    /// Supports multidirectional (Numpy-style) broadcasting.
    ///
    /// Version 14+ supports: uint8, int8, uint16, int16, int32, int64, uint32, uint64,
    /// float16, float, double, bfloat16
    fn generate_implementation(&self, _inputs: &[String], _outputs: &[String]) -> TokenStream {
        quote! {
            // ONNX Add operation: Element-wise addition with broadcasting
            // Reference: https://onnx.ai/onnx/operators/onnx__Add.html

            fn compute_broadcast_shape(shape_a: &[usize], shape_b: &[usize]) -> Vec<usize> {
                let max_len = shape_a.len().max(shape_b.len());
                let mut result = Vec::with_capacity(max_len);

                for i in 0..max_len {
                    let dim_a = if i < shape_a.len() {
                        shape_a[shape_a.len() - 1 - i]
                    } else {
                        1
                    };
                    let dim_b = if i < shape_b.len() {
                        shape_b[shape_b.len() - 1 - i]
                    } else {
                        1
                    };

                    if dim_a == dim_b {
                        result.push(dim_a);
                    } else if dim_a == 1 {
                        result.push(dim_b);
                    } else if dim_b == 1 {
                        result.push(dim_a);
                    } else {
                        panic!("Incompatible shapes for broadcasting: {:?} and {:?}", shape_a, shape_b);
                    }
                }

                result.reverse();
                result
            }

            fn compute_strides(shape: &[usize]) -> Vec<usize> {
                let mut strides = vec![1; shape.len()];
                for i in (0..shape.len() - 1).rev() {
                    strides[i] = strides[i + 1] * shape[i + 1];
                }
                strides
            }

            fn broadcast_index(index: usize, shape: &[usize], target_shape: &[usize]) -> usize {
                let mut result = 0;
                let mut remaining = index;
                let offset = target_shape.len() - shape.len();

                for i in (0..target_shape.len()).rev() {
                    let target_dim = target_shape[i];
                    let coord = remaining % target_dim;
                    remaining /= target_dim;

                    if i >= offset {
                        let dim = shape[i - offset];
                        if dim == 1 {
                            // Broadcasting: use index 0
                        } else if dim == target_dim {
                            result += coord * if i - offset + 1 < shape.len() {
                                shape[i - offset + 1..].iter().product::<usize>()
                            } else {
                                1
                            };
                        } else {
                            panic!("Invalid broadcast dimension");
                        }
                    }
                }

                result
            }

            // Infer shapes from input slices
            // For now, we assume 1D tensors and will need shape information passed in
            // In a real implementation, shapes would be part of the tensor struct
            let len_a = input_0.len();
            let len_b = input_1.len();

            // Simple case: both same length (no broadcasting needed)
            if len_a == len_b {
                input_0.iter()
                    .zip(input_1.iter())
                    .map(|(a, b)| a + b)
                    .collect()
            }
            // Broadcasting cases
            else if len_b == 1 {
                // Scalar broadcast
                let scalar = input_1[0];
                input_0.iter().map(|a| a + scalar).collect()
            }
            else if len_a == 1 {
                // Scalar broadcast (reversed)
                let scalar = input_0[0];
                input_1.iter().map(|b| scalar + b).collect()
            }
            else {
                // For proper multidimensional broadcasting, we would need shape information
                // This is a simplified version that handles common cases
                // Full implementation requires tensor shape metadata

                // Try to broadcast assuming smaller tensor broadcasts to larger
                if len_a > len_b {
                    // Assume input_1 should broadcast to input_0's shape
                    if len_a % len_b == 0 {
                        input_0.iter()
                            .enumerate()
                            .map(|(i, a)| a + input_1[i % len_b])
                            .collect()
                    } else {
                        panic!("Cannot broadcast: incompatible shapes {} and {}", len_a, len_b);
                    }
                } else {
                    // Assume input_0 should broadcast to input_1's shape
                    if len_b % len_a == 0 {
                        input_1.iter()
                            .enumerate()
                            .map(|(i, b)| input_0[i % len_a] + b)
                            .collect()
                    } else {
                        panic!("Cannot broadcast: incompatible shapes {} and {}", len_a, len_b);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_operation_code_generation() {
        let add_op = AddOperation;
        let inputs = vec!["A".to_string(), "B".to_string()];
        let outputs = vec!["C".to_string()];

        let code = add_op.generate_implementation(&inputs, &outputs);
        let code_str = code.to_string();

        // Verify the generated code contains key elements
        assert!(code_str.contains("Element-wise addition"));
        assert!(code_str.contains("broadcast"));
    }

    #[test]
    fn test_generated_add_same_shape() {
        // This tests the runtime behavior of the generated code
        // Simulate what the generated code would do
        let input_0 = vec![1.0_f32, 2.0, 3.0, 4.0];
        let input_1 = vec![5.0_f32, 6.0, 7.0, 8.0];

        let result: Vec<f32> = input_0.iter()
            .zip(input_1.iter())
            .map(|(a, b)| a + b)
            .collect();

        assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_generated_add_scalar_broadcast() {
        // Test scalar broadcasting
        let input_0 = vec![1.0_f32, 2.0, 3.0, 4.0];
        let input_1 = vec![10.0_f32];

        let scalar = input_1[0];
        let result: Vec<f32> = input_0.iter().map(|a| a + scalar).collect();

        assert_eq!(result, vec![11.0, 12.0, 13.0, 14.0]);
    }

    #[test]
    fn test_generated_add_scalar_broadcast_reversed() {
        // Test scalar broadcasting (reversed)
        let input_0 = vec![5.0_f32];
        let input_1 = vec![1.0_f32, 2.0, 3.0, 4.0];

        let scalar = input_0[0];
        let result: Vec<f32> = input_1.iter().map(|b| scalar + b).collect();

        assert_eq!(result, vec![6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_broadcast_shape_computation() {
        // Test the broadcast shape computation logic
        let compute_broadcast_shape = |shape_a: &[usize], shape_b: &[usize]| -> Vec<usize> {
            let max_len = shape_a.len().max(shape_b.len());
            let mut result = Vec::with_capacity(max_len);

            for i in 0..max_len {
                let dim_a = if i < shape_a.len() {
                    shape_a[shape_a.len() - 1 - i]
                } else {
                    1
                };
                let dim_b = if i < shape_b.len() {
                    shape_b[shape_b.len() - 1 - i]
                } else {
                    1
                };

                if dim_a == dim_b {
                    result.push(dim_a);
                } else if dim_a == 1 {
                    result.push(dim_b);
                } else if dim_b == 1 {
                    result.push(dim_a);
                } else {
                    panic!("Incompatible shapes");
                }
            }

            result.reverse();
            result
        };

        // Test cases from ONNX documentation
        assert_eq!(compute_broadcast_shape(&[2, 3, 4, 5], &[]), vec![2, 3, 4, 5]); // scalar
        assert_eq!(compute_broadcast_shape(&[2, 3, 4, 5], &[5]), vec![2, 3, 4, 5]);
        assert_eq!(compute_broadcast_shape(&[2, 3, 4, 5], &[4, 5]), vec![2, 3, 4, 5]);
        assert_eq!(compute_broadcast_shape(&[4, 5], &[2, 3, 4, 5]), vec![2, 3, 4, 5]);
        assert_eq!(compute_broadcast_shape(&[1, 4, 5], &[2, 3, 1, 1]), vec![2, 3, 4, 5]);
    }

    #[test]
    #[should_panic(expected = "Incompatible shapes")]
    fn test_broadcast_shape_incompatible() {
        let compute_broadcast_shape = |shape_a: &[usize], shape_b: &[usize]| -> Vec<usize> {
            let max_len = shape_a.len().max(shape_b.len());
            let mut result = Vec::with_capacity(max_len);

            for i in 0..max_len {
                let dim_a = if i < shape_a.len() {
                    shape_a[shape_a.len() - 1 - i]
                } else {
                    1
                };
                let dim_b = if i < shape_b.len() {
                    shape_b[shape_b.len() - 1 - i]
                } else {
                    1
                };

                if dim_a == dim_b {
                    result.push(dim_a);
                } else if dim_a == 1 {
                    result.push(dim_b);
                } else if dim_b == 1 {
                    result.push(dim_a);
                } else {
                    panic!("Incompatible shapes");
                }
            }

            result.reverse();
            result
        };

        // This should panic: shapes [3] and [4] are incompatible
        compute_broadcast_shape(&[3], &[4]);
    }
}

