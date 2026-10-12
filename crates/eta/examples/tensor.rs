//! Construct a dense CPU tensor through the public facade.

use eta::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let tensor = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2])?;
  println!("shape: {:?}, data: {:?}", tensor.shape(), tensor.as_slice());
  Ok(())
}
