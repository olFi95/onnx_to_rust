use quote::__private::TokenStream;
use quote::quote;
use super::OperationCodeGenerator;

pub struct AddOperation;

impl OperationCodeGenerator for AddOperation {
    /// https://onnx.ai/onnx/operators/onnx__Add.html
    fn generate_implementation(&self, _inputs: &[String], _outputs: &[String]) -> TokenStream {
        quote! {
            // TODO: Implement Add operation
            // Element-wise addition of two tensors
            unimplemented!("Add operation not yet implemented")
        }
    }
}

