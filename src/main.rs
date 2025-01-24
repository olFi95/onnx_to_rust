use std::fmt::Display;
use prost::Message;
use std::fs::File;
use std::io::Read;

include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
fn deserialize_protobuf_file(file_path: &str) -> Result<ModelProto, Box<dyn std::error::Error>> {
    // Open the file
    let mut file = File::open(file_path)?;

    // Read the file content into a byte buffer
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    // Deserialize the buffer into a Protobuf message
    let model_proto = ModelProto::decode(&*buffer)?;

    Ok(model_proto)
}

fn main() {
    let model_proto = deserialize_protobuf_file("resources/mnist-8.onnx").expect("cannot deserialize .onnx file");
    print_metadata(model_proto);

}

fn print_metadata(model_proto: ModelProto) {
    match model_proto.model_version {
        Some(version) => {println!("Model version: {}", version);}
        None => {println!("Model version not found");}
    }
    match model_proto.producer_name {
        Some(producer) => {println!("Model was produced by: {}", producer);}
        None => {println!("Model was produced by unknown");}
    }
    match model_proto.domain {
        Some(domain) => {println!("Domain: {}", domain);}
        None => {println!("Model has no Domain");}
    }
    match model_proto.doc_string {
        Some(docstring) => {println!("Docstring: {}", docstring);}
        None => {println!("Model has no Docstring");}
    }
    for metadata_prop in model_proto.metadata_props {
        match (metadata_prop.key, metadata_prop.value) {
            (Some(key), Some(value)) => {println!("Key: {key} Value: {value}");}
            (Some(key), None) => {println!("Key: {key}");}
            (None, Some(value)) => {println!("Value: {value}");}
            (None, None) => {}
        }
    }

    for opset_import in model_proto.opset_import {
        match opset_import.domain {
            Some(domain) => {println!("Opset Import Domain: {}", domain);}
            None => {println!("Opset Import has no Domain");}
        }
        match opset_import.version {
            Some(version) => {println!("Opset Import Version: {}", version);}
            None => {println!("Opset Import version not found");}
        }
    }

    for function in model_proto.functions {
        println!("{:?}", function);
    }

    for training_info in model_proto.training_info {
        match training_info.algorithm {
            Some(algorithm) => {println!("Training Algorithm: {:?}", algorithm);}
            None => {}
        }
        match training_info.initialization {
            Some(initialization) => {println!("Training initialisation: {:?}", initialization);}
            None => {}
        }
        for initialisation_binding in training_info.initialization_binding{
            println!("Training initialization_binding: {:?}", initialisation_binding);
        }
        for update_binding in  training_info.update_binding{
            println!("Training update_binding: {:?}", update_binding);
        }
    }

    match model_proto.graph {
        Some(graph) => {
            println!("NODES:");
            for node in graph.node {
                println!("Node name: {}", node.name.unwrap());
                println!("Node op_type: {}", node.op_type.unwrap());
                println!("Inputs: ");
                for input in node.input {
                    println!("Input: {}", input);
                }
                println!("Outputs: ");
                for output in node.output {
                    println!("Outputs: {}", output);
                }
            }
        }
        _ => {}
    }
}
