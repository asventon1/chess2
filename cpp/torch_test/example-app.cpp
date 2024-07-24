#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>

int main() {
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;
  auto model = torch::jit::load("/home/adam/stuff/python/chess2/stockfish_model_small.pt");
  model.eval();
  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0);
  std::vector<torch::jit::IValue> thing = {torch::zeros({833}, options)};
  auto prediction = model.forward(thing);
  std::cout << prediction << std::endl;
}
