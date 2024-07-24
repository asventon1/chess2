#include <torch/torch.h>
#include <torch/script.h>
#include "../include/main.h"
#include "../chess-library/include/chess.hpp"
#include <string>
#include <sstream>

using namespace chess;

std::vector<std::string> get_words(std::string s){
    std::vector<std::string> res;
    int pos = 0;
    for(int i = 0; i < 2; i++){
        pos = s.find(" ");
        res.push_back(s.substr(0,pos));
        s.erase(0,pos+1);
   }
    return res;
}

std::array<float, 64*13+1> board_array_from_fen(std::string fen_input) {
  std::array<float, 64*13+1> current_board;
  for(unsigned int i = 0; i < 64; i++) {
    current_board[i*13] = 1;
    for(unsigned int j = 0; j < 12; j++){
      current_board[i*13+j+1] = 0;
    }
  }
  std::vector<std::string> whole_fen = get_words(fen_input);
  std::string fen = whole_fen[0];
  std::string color = whole_fen[1];
  char current_color = color == "w" ? 1 : 0;
  unsigned int square = 0;
  for(unsigned int i = 0; i < fen.length(); i++) {
    char c = fen[i];
    if(c == '/') {
    } else if(isalpha(c)) {
      current_board[square*13+0] = 0;
      if(c == 'p') { current_board[square*13+1] = 1; }
      else if(c == 'n') { current_board[square*13+2] = 1; }
      else if(c == 'b') { current_board[square*13+3] = 1; }
      else if(c == 'r') { current_board[square*13+4] = 1; }
      else if(c == 'q') { current_board[square*13+5] = 1; }
      else if(c == 'k') { current_board[square*13+6] = 1; }
      else if(c == 'P') { current_board[square*13+7] = 1; }
      else if(c == 'N') { current_board[square*13+8] = 1; }
      else if(c == 'B') { current_board[square*13+9] = 1; }
      else if(c == 'R') { current_board[square*13+10] = 1; }
      else if(c == 'Q') { current_board[square*13+11] = 1; }
      else if(c == 'K') { current_board[square*13+12] = 1; }
      square += 1; 
    } else {
      square += (c - '0');
    }
  }
  current_board[64*13] = current_color;
  return current_board;
}

/*
fn board_array_from_fen(fen_input: &str) -> Vec<u8> {
    let mut current_board = vec![[1,0,0,0,0,0,0,0,0,0,0,0,0]; 64];
    let whole_fen = fen_input.split_whitespace().collect::<Vec<&str>>();
    let fen = whole_fen[0];
    let color = whole_fen[1];
    let current_color = if color == "w" { 1 } else { 0 };
    let mut square = 0;
    for c in fen.chars(){
        if c == '/' {
        } else if c.is_alphabetic() {
            current_board[square][0] = 0;
            if c == 'p' { current_board[square][1] = 1 }
            else if c == 'n' { current_board[square][2] = 1 }
            else if c == 'b' { current_board[square][3] = 1 }
            else if c == 'r' { current_board[square][4] = 1 }
            else if c == 'q' { current_board[square][5] = 1 }
            else if c == 'k' { current_board[square][6] = 1 }
            else if c == 'P' { current_board[square][7] = 1 }
            else if c == 'N' { current_board[square][8] = 1 }
            else if c == 'B' { current_board[square][9] = 1 }
            else if c == 'R' { current_board[square][10] = 1 }
            else if c == 'Q' { current_board[square][11] = 1 }
            else if c == 'K' { current_board[square][12] = 1 }
            square += 1;
        } else {
            square += c.to_digit(10).unwrap() as usize;
        }
    }
    let mut current_board = current_board.into_iter().flatten().collect::<Vec<u8>>();
    current_board.push(current_color);
    return current_board;
}
*/

torch::jit::script::Module model;
torch::Device device("cpu:0");

float get_board_value(Board board) {
  torch::NoGradGuard no_grad;
  auto options = torch::TensorOptions().dtype(torch::kFloat32);
  std::array<float, 833> array_data = board_array_from_fen(board.getFen());
  torch::Tensor tensor_data = torch::from_blob(array_data.data(), {833}, options);
  tensor_data = tensor_data.to(device);
  std::vector<torch::jit::IValue> input_data = {tensor_data};
  auto prediction = model.forward(input_data);
  float pred_float = prediction.toTensor().item<float>();
  //std::cout << pred_float << std::endl;
  return pred_float;
  //return 1;
}

std::pair<float, std::optional<Move>> minimax_internal(Board board, int depth, double alpha, double beta, int color) {
  std::optional<Move> best_move = std::nullopt;
  Movelist moves;
  movegen::legalmoves(moves, board);
  if(moves.empty()){
    if(board.inCheck()){
      if(board.sideToMove() == Color::WHITE) return std::make_pair(-1000, std::nullopt);
      else return std::make_pair(1000, std::nullopt);
    } else {
      return std::make_pair(0.5, std::nullopt);
    }
  }
  if(board.isInsufficientMaterial() || board.isHalfMoveDraw() || board.isRepetition(3)){
      return std::make_pair(0.5, std::nullopt);
  }
/*
  GameResult outcome = board.isGameOver().second;
  switch(outcome) {
    case GameResult::NONE:
      break;
    case GameResult::WIN:
    case GameResult::LOSE:
    case GameResult::DRAW:
      std::cout << "test" << std::endl;
      return std::make_pair(100, std::nullopt);
  }
  */
  if(depth == 0) {
    float output = color * get_board_value(board);
    return std::make_pair(color*get_board_value(board), std::nullopt);
  }
  float best_score = -10000000;
  for(const auto &m : moves) {
    board.makeMove(m);
    std::pair<float, std::optional<Move>> minimax_result = minimax_internal(board, depth-1, -beta, -alpha, -color);
    float score = minimax_result.first;
    std::optional<Move> move = minimax_result.second;
    score = -score;
    if(score > best_score) { 
      best_score = score;
      best_move = m;
    }
    if(best_score > alpha) {
      alpha = best_score;
    }
    board.unmakeMove(m);
    if(alpha >= beta) {
      break;
    }
  }
  return std::make_pair(best_score, best_move);
}

std::string minimax(const std::string &fen, unsigned int depth){

  model = torch::jit::load("/home/adam/stuff/python/chess2/stockfish_model_small.pt");
  model.eval();
  //device = torch::Device("cpu:0");
  model.to(device);
  Board board = Board(fen);
  int color = board.sideToMove() == Color::WHITE ? 1 : -1;

  std::pair<int, std::optional<Move>> minimax_result = minimax_internal(board, depth, -100000000, 100000000, color);
  int best_score = minimax_result.first;
  Move best_move = minimax_result.second.value();

  std::stringstream buffer;
  buffer << best_move << std::endl;
  std::string output = buffer.str();
  std::cout << output << std::endl;

  return output;
}

int main() {
  minimax("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 4);
}
