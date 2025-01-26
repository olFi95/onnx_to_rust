use crate::ModelProto;
use num::Num;
use quote::__private::TokenStream;
use quote::{format_ident, quote, ToTokens, TokenStreamExt};
use std::fmt::Display;
use std::path::PathBuf;

/// I use only the basic types from https://github.com/onnx/onnx/blob/main/onnx/onnx.proto#L504
pub enum TensorProtoDataType {
    UNDEFINED = 0,
    // Basic types.
    FLOAT = 1,  // float
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
    output_file: PathBuf
}


impl<'a> OnnxCodeGenerator<'a> {
    pub(crate) fn new(model_proto: &'a ModelProto, output_file: PathBuf) -> Self {
        OnnxCodeGenerator{model_proto, output_file}
    }
    fn set_output_file(&mut self, output_file: PathBuf) {
        self.output_file = output_file;
    }
    // pub fn find_variable_definition<'a>(&self, var_name: &str) -> &'a Option<TensorProto> {
    //     self.graph_proto.initializer.iter().as_ref().find(|i| { i.name.unwrap() == var_name })
    // }

    pub fn generate_tensor_data(&self) -> TokenStream {
        let mut output = quote! {};
        for tensor in self.model_proto.graph.as_ref().unwrap().initializer.clone(){
            let tensor_name = format_ident!("{}", tensor.name.expect("Tensor name missing"));
            println!("{tensor_name}");
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
        result.push('['); // Start of the current array level
        if depth == 0 {
            result.push_str(datatype); // The innermost dimension is always i32
        } else {
            result.push_str(&generate_recursive(dimensions, depth-1, datatype));
        }
        result.push_str("; ");
        result.push_str(dimensions[depth].to_string().as_str());

        result.push(']'); // End of the current array level

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


#[cfg(test)]
mod tests {
    // Bring the outer scope into the test module
    use super::*;

    // Unit test for the `add` function
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