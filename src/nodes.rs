// Code generator for ONNX models
use crate::ModelProto;
use num::Num;
use quote::__private::TokenStream;
use quote::{format_ident, quote, TokenStreamExt};
use std::fmt::Display;

/// Basic ONNX tensor data types
/// Reference: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto#L504
pub enum TensorProtoDataType {
    UNDEFINED = 0,
    // Basic types
    FLOAT = 1,   // float
    UINT8 = 2,   // uint8_t
    INT8 = 3,    // int8_t
    UINT16 = 4,  // uint16_t
    INT16 = 5,   // int16_t
    INT32 = 6,   // int32_t
    INT64 = 7,   // int64_t
    STRING = 8,  // string
    BOOL = 9,    // bool
}
impl TensorProtoDataType {
}


pub struct OnnxCodeGenerator<'a> {
    model_proto: &'a ModelProto,
}


impl<'a> OnnxCodeGenerator<'a> {
    pub(crate) fn new(model_proto: &'a ModelProto) -> Self {
        OnnxCodeGenerator{model_proto}
    }

    /// Generates all inference methods for the layers/nodes of the ONNX model
    pub fn generate_inference_methods(&self) -> TokenStream {
        let mut output = quote! {};

        let graph = self.model_proto.graph.as_ref().expect("Model has no graph");

        // Generate individual layer methods
        for node in &graph.node {
            let node_name = node.name.as_ref().expect("Node has no name");
            let op_type = node.op_type.as_ref().expect("Node has no op_type");

            // Generate method name from node name (sanitize for valid Rust identifiers)
            let method_name = format_ident!("layer_{}", sanitize_identifier(node_name));

            // Collect input parameters
            let input_params = self.generate_input_parameters(&node.input);

            // Determine output types
            let output_type = self.generate_output_type(&node.output);

            // Generate operation stub using operations modules
            let implementation = self.generate_operation_stub(op_type, &node.input, &node.output);

            // Doc comments with actual values
            let doc_comment = format!("Inference method for node: {}\nOperation: {}", node_name, op_type);

            output.append_all(quote! {
                #[doc = #doc_comment]
                pub fn #method_name(#input_params) -> #output_type {
                    #implementation
                }
            });
        }

        // Generate main inference method that chains all layers
        let main_inference = self.generate_main_inference_method();
        output.append_all(main_inference);

        output
    }

    /// Generates the main inference method that executes all layers in sequence
    pub fn generate_main_inference_method(&self) -> TokenStream {
        let graph = self.model_proto.graph.as_ref().expect("Model has no graph");

        // Collect all initializer names (these are static tensors)
        let initializer_names: std::collections::HashSet<String> = graph.initializer.iter()
            .filter_map(|init| init.name.clone())
            .collect();

        // Collect graph inputs (filter out initializers - only keep real inputs)
        let all_inputs: Vec<String> = graph.input.iter()
            .filter_map(|input| input.name.clone())
            .collect();

        let real_inputs: Vec<String> = all_inputs.iter()
            .filter(|name| !initializer_names.contains(*name))
            .cloned()
            .collect();

        // Collect graph outputs
        let graph_outputs: Vec<String> = graph.output.iter()
            .filter_map(|output| output.name.clone())
            .collect();

        // Generate the layer execution code
        let mut layer_executions = quote! {};

        // Create a HashMap to store intermediate results
        layer_executions.append_all(quote! {
            use std::collections::HashMap;
            let mut intermediate_outputs: HashMap<String, Vec<f32>> = HashMap::new();
        });

        // Add only real inputs (not initializers) to intermediate outputs
        for (idx, input_name) in real_inputs.iter().enumerate() {
            let param_name = format_ident!("input_{}", idx);
            layer_executions.append_all(quote! {
                intermediate_outputs.insert(#input_name.to_string(), #param_name.to_vec());
            });
        }

        // Execute each layer in sequence
        for node in &graph.node {
            let node_name = node.name.as_ref().expect("Node has no name");
            let method_name = format_ident!("layer_{}", sanitize_identifier(node_name));

            // Generate code to retrieve inputs for this layer
            let mut input_vars = Vec::new();
            let mut input_retrieval = quote! {};

            for (idx, input_name) in node.input.iter().enumerate() {
                let var_name = format_ident!("input_{}_for_{}", idx, sanitize_identifier(node_name));
                input_vars.push(var_name.clone());

                // Check if it's an initializer (static tensor) or intermediate output
                if initializer_names.contains(input_name) {
                    // It's a static tensor - flatten and use it directly
                    let tensor_name = format_ident!("{}", input_name);

                    // Check if it's an i64 tensor (for shapes)
                    if input_name.contains("shape") {
                        input_retrieval.append_all(quote! {
                            let #var_name: Vec<f32> = #tensor_name.iter()
                                .copied()
                                .map(|x| x as f32)
                                .collect();
                        });
                    } else {
                        // Regular f32 tensor - flatten all dimensions
                        input_retrieval.append_all(quote! {
                            let #var_name: Vec<f32> = #tensor_name.iter()
                                .flat_map(|a| a.iter())
                                .flat_map(|b| b.iter())
                                .flat_map(|c| c.iter())
                                .flat_map(|d| d.iter())
                                .copied()
                                .collect();
                        });
                    }
                } else {
                    // It's an intermediate output or input - get from HashMap
                    input_retrieval.append_all(quote! {
                        let #var_name = intermediate_outputs
                            .get(#input_name)
                            .expect(&format!("Input '{}' not found", #input_name))
                            .clone();
                    });
                }
            }

            // Generate the layer call with all inputs as slices
            let input_refs: Vec<_> = input_vars.iter()
                .map(|var| quote! { #var.as_slice() })
                .collect();

            // Store the output(s)
            if node.output.len() == 1 {
                let output_name = &node.output[0];
                layer_executions.append_all(quote! {
                    #input_retrieval
                    let output = #method_name(#(#input_refs),*);
                    intermediate_outputs.insert(#output_name.to_string(), output);
                });
            } else {
                // Multiple outputs (tuple)
                let output_names = &node.output;
                let output_vars: Vec<_> = (0..output_names.len())
                    .map(|i| format_ident!("out_{}", i))
                    .collect();

                layer_executions.append_all(quote! {
                    #input_retrieval
                    let (#(#output_vars),*) = #method_name(#(#input_refs),*);
                });

                for (idx, output_name) in output_names.iter().enumerate() {
                    let var = &output_vars[idx];
                    layer_executions.append_all(quote! {
                        intermediate_outputs.insert(#output_name.to_string(), #var);
                    });
                }
            }
        }

        // Return the final output(s)
        let return_statement = if graph_outputs.len() == 1 {
            let output_name = &graph_outputs[0];
            quote! {
                intermediate_outputs.remove(#output_name).expect("Output not found")
            }
        } else {
            let output_retrievals: Vec<_> = graph_outputs.iter()
                .map(|name| quote! {
                    intermediate_outputs.remove(#name).expect("Output not found")
                })
                .collect();
            quote! {
                (#(#output_retrievals),*)
            }
        };

        // Generate input parameters for main inference (only real inputs, not initializers)
        let mut main_input_params = quote! {};
        for (idx, _) in real_inputs.iter().enumerate() {
            let param_name = format_ident!("input_{}", idx);
            main_input_params.append_all(quote! {
                #param_name: &[f32],
            });
        }

        // Determine return type
        let return_type = if graph_outputs.len() == 1 {
            quote! { Vec<f32> }
        } else {
            let output_types = (0..graph_outputs.len()).map(|_| quote! { Vec<f32> });
            quote! { (#(#output_types),*) }
        };

        quote! {
            /// Main inference method that executes the entire model
            ///
            /// This method chains all layers in the correct order according to the ONNX graph,
            /// passing intermediate results between layers.
            pub fn infer(#main_input_params) -> #return_type {
                #layer_executions
                #return_statement
            }
        }
    }

    /// Generiert die Input-Parameter für eine Methode
    fn generate_input_parameters(&self, inputs: &[String]) -> TokenStream {
        let mut params = quote! {};

        for (idx, _input_name) in inputs.iter().enumerate() {
            let param_name = format_ident!("input_{}", idx);
            // Dummy-Typ - später können wir die echten Typen aus dem Graph ermitteln
            params.append_all(quote! {
                #param_name: &[f32],
            });
        }

        params
    }

    /// Generates the output type for a method
    fn generate_output_type(&self, outputs: &[String]) -> TokenStream {
        if outputs.len() == 1 {
            quote! { Vec<f32> }
        } else {
            // Multiple outputs as tuple
            let output_types = (0..outputs.len()).map(|_| quote! { Vec<f32> });
            quote! { (#(#output_types),*) }
        }
    }

    /// Generates a stub for the operation using match statement
    /// Delegates to operation-specific modules for code generation
    fn generate_operation_stub(&self, op_type: &str, inputs: &[String], outputs: &[String]) -> TokenStream {
        use crate::operations::OperationCodeGenerator;

        match op_type {
            "Gemm" => crate::operations::gemm::GemmOperation.generate_implementation(inputs, outputs),
            "Relu" => crate::operations::relu::ReluOperation.generate_implementation(inputs, outputs),
            "Softmax" => crate::operations::softmax::SoftmaxOperation.generate_implementation(inputs, outputs),
            "Conv" => crate::operations::conv::ConvOperation.generate_implementation(inputs, outputs),
            "MaxPool" => crate::operations::maxpool::MaxPoolOperation.generate_implementation(inputs, outputs),
            "Add" => crate::operations::add::AddOperation.generate_implementation(inputs, outputs),
            "Mul" => crate::operations::mul::MulOperation.generate_implementation(inputs, outputs),
            "MatMul" => crate::operations::matmul::MatMulOperation.generate_implementation(inputs, outputs),
            "Reshape" => crate::operations::reshape::ReshapeOperation.generate_implementation(inputs, outputs),
            "Flatten" => crate::operations::flatten::FlattenOperation.generate_implementation(inputs, outputs),
            "BatchNormalization" => crate::operations::batch_normalization::BatchNormalizationOperation.generate_implementation(inputs, outputs),
            "Dropout" => crate::operations::dropout::DropoutOperation.generate_implementation(inputs, outputs),
            _ => {
                // Fallback for unsupported operations
                quote! {
                    unimplemented!(concat!("Operation ", #op_type, " not yet implemented"))
                }
            }
        }
    }

    pub fn generate_tensor_data(&self) -> TokenStream {
        let mut output = quote! {};
        for tensor in self.model_proto.graph.as_ref().unwrap().initializer.clone(){
            let tensor_name = format_ident!("{}", tensor.name.expect("Tensor name missing"));
            let tensor_datatype_id = tensor.data_type.expect("Tensor data-type missing");
            let tensor_datatype_onnx = Self::from_i32(tensor_datatype_id).expect("no onnx type found for id");
            let tensor_datatype_rust = format_ident!("{}", rust_type(&tensor_datatype_onnx));

            let tensor_dimensionality = generate_array_declaration_string(&tensor.dims, tensor_datatype_rust.to_string().as_str());
            let tensor_dimensionality_tokens: proc_macro2::TokenStream = tensor_dimensionality.parse().unwrap();
            let tensor_data_string = match Self::from_i32(tensor_datatype_id) {
                Some(TensorProtoDataType::FLOAT) => generate_array_data_string(&tensor.dims, &tensor.float_data),
                Some(TensorProtoDataType::INT64) => generate_array_data_string(&tensor.dims, &tensor.int64_data),
                None => {panic!("unsupported datatype id")}
                _ => {panic!("unsupported datatype id")}
            };
            let tensor_data_tokens: proc_macro2::TokenStream = tensor_data_string.parse().unwrap();
            output.append_all(quote! {
                pub static #tensor_name: #tensor_dimensionality_tokens  = #tensor_data_tokens;
            });
        }
        output
    }
    pub fn from_i32(value: i32) -> Option<TensorProtoDataType> {
        match value {
            0 => Some(TensorProtoDataType::UNDEFINED),
            1 => Some(TensorProtoDataType::FLOAT),
            2 => Some(TensorProtoDataType::UINT8),
            3 => Some(TensorProtoDataType::INT8),
            4 => Some(TensorProtoDataType::UINT16),
            5 => Some(TensorProtoDataType::INT16),
            6 => Some(TensorProtoDataType::INT32),
            7 => Some(TensorProtoDataType::INT64),
            8 => Some(TensorProtoDataType::STRING),
            9 => Some(TensorProtoDataType::BOOL),
            _ => None,
        }
    }

}

pub fn rust_type(onnx_type: &TensorProtoDataType) -> String {
    match *onnx_type {
        TensorProtoDataType::UNDEFINED => "undefined".to_string(),
        TensorProtoDataType::FLOAT => "f32".to_string(),
        TensorProtoDataType::UINT8 => "u8".to_string(),
        TensorProtoDataType::INT8 => "i8".to_string(),
        TensorProtoDataType::UINT16 => "u16".to_string(),
        TensorProtoDataType::INT16 => "i16".to_string(),
        TensorProtoDataType::INT32 => "i32".to_string(),
        TensorProtoDataType::INT64 => "i64".to_string(),
        TensorProtoDataType::STRING => "&str".to_string(),
        TensorProtoDataType::BOOL => "bool".to_string(),
    }
}
fn generate_array_declaration_string(dimensions: &Vec<i64>, datatype: &str) -> String {
    fn generate_recursive(dimensions: &Vec<i64>, depth: usize, datatype: &str) -> String {
        let mut result = String::new();
        result.push('[');
        if depth == 0 {
            result.push_str(datatype);
        } else {
            result.push_str(&generate_recursive(dimensions, depth-1, datatype));
        }
        result.push_str("; ");
        result.push_str(dimensions[depth].to_string().as_str());
        result.push(']');
        result
    }

    // Call the recursive function starting from depth 0
    generate_recursive(dimensions, dimensions.len()-1, datatype)
}
fn generate_array_data_string<T: Num + Copy + std::fmt::Display>(dimensions: &Vec<i64>, x: &Vec<T>) -> String {
    fn recurse<T: Num + Copy + Display>(dimensions: &[i64], x: &[T]) -> String {
        let mut result = String::new();
        result.push_str("[");
        if dimensions.len() > 1 {
            for i in 0..dimensions[0] as usize {
                let block_size:i64 = dimensions[1..dimensions.len()].iter().product();
                let start_point = i * block_size as usize;
                let end_point = start_point + block_size as usize ;
                result.push_str(recurse(&dimensions[1..dimensions.len()], &x[start_point..end_point]).as_str());
                if i < dimensions[0] as usize -1{
                    result.push_str(", ");
                }
            }
        } else {
            for i in 0..dimensions[0] {
                result.push_str(format!("{:.16}",x[i as usize]).as_str());
                if i < dimensions[0] - 1 {
                    result.push_str(", ")
                }
            }
        }
        result.push_str("]");
        result
    }
    return recurse(&dimensions, &x)
}

/// Sanitizes a string to make it a valid Rust identifier
fn sanitize_identifier(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' {
                c.to_lowercase().next().unwrap_or(c)
            } else {
                '_'
            }
        })
        .collect()
}


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_generate_array_data_string() {
        let dimensions = vec![2, 4];
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let data_array = generate_array_data_string(&dimensions, &data);
        assert_eq!(data_array, "[[1.0000000000000000, 2.0000000000000000, 3.0000000000000000, 4.0000000000000000], [5.0000000000000000, 6.0000000000000000, 7.0000000000000000, 8.0000000000000000]]");
    }

    #[test]
    fn test_generate_array_data_string_1d_input() {
        let dimensions = vec![2];
        let data: Vec<f32> = vec![1.0, 2.0];
        let data_array = generate_array_data_string(&dimensions, &data);
        assert_eq!(data_array, "[1.0000000000000000, 2.0000000000000000]");
    }

    #[test]
    fn test() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let other_array = &data[0..data.len()];
        println!("data: {:?}", data);
        println!("other_array: {:?}", other_array);
    }
}