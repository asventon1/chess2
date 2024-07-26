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

float get_board_value_pieces(Board board) {
  int score = 0;
  for(int i = 0; i < 64; i++) {
    Piece piece = board.at(Square(i));
    if(piece == Piece::WHITEPAWN) {
      score += 1;
    } else if(piece == Piece::WHITEKNIGHT) {
      score += 3;
    } else if(piece == Piece::WHITEBISHOP) {
      score += 3;
    } else if(piece == Piece::WHITEROOK) {
      score += 5;
    } else if(piece == Piece::WHITEQUEEN) {
      score += 9;
    } else if(piece == Piece::WHITEKING) {
      score += 100;
    } else if(piece == Piece::BLACKPAWN) {
      score -= 1;
    } else if(piece == Piece::BLACKKNIGHT) {
      score -= 3;
    } else if(piece == Piece::BLACKBISHOP) {
      score -= 3;
    } else if(piece == Piece::BLACKROOK) {
      score -= 5;
    } else if(piece == Piece::BLACKQUEEN) {
      score -= 9;
    } else if(piece == Piece::BLACKKING) {
      score -= 100;
    }
  }
  return score;
}

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
  float piece_value = get_board_value_pieces(board);
  return pred_float + piece_value/2.0;
  /*
  float piece_value = get_board_value_pieces(board);
  */
  //return piece_value;
}

float extra_depth = 0;

std::pair<float, std::optional<Move>> minimax_internal(Board board, float depth, double alpha, double beta, int color) {
  std::optional<Move> best_move = std::nullopt;
  Movelist moves;
  movegen::legalmoves(moves, board);
  if(moves.empty()){
    if(board.inCheck()){
      if(board.sideToMove() == Color::WHITE){
        return std::make_pair(-1000.0*(depth+1), std::nullopt);
      } else {
        return std::make_pair(-1000.0*(depth+1), std::nullopt);
      }
    } else {
      return std::make_pair(0.5, std::nullopt);
    }
  }
  if(board.isInsufficientMaterial() || board.isHalfMoveDraw() || board.isRepetition()){
    //std::cout << "test" << std::endl; 
    return std::make_pair(0.5, std::nullopt);
  }
  if(depth < 0.00001) {
    float output = color * get_board_value(board);
    return std::make_pair(color*get_board_value(board), std::nullopt);
  }
  float best_score = -10000000;
  for(const auto &m : moves) {
    float current_depth = depth;
    if(board.isCapture(m) || board.inCheck()){
      current_depth += extra_depth;
    }
    board.makeMove(m);
    std::pair<float, std::optional<Move>> minimax_result = minimax_internal(board, current_depth-1, -beta, -alpha, -color);
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

struct ttTableEntry {
  int value;
  std::string flag;
  int depth;
  ttTableEntry(int value, std::string flag, int depth) {
    this->value = value;
    this->flag = flag;
    this->depth = depth;
  }
};

std::optional<Board> board = std::nullopt;

std::string minimax(const std::string &fen, const std::string &move, unsigned int depth, float extra_depth_i){

  extra_depth = extra_depth_i;

  torch::set_num_threads(24);

  model = torch::jit::load("/home/adam/stuff/python/chess2/stockfish_model_small3.pt");
  model.eval();
  //device = torch::Device("cpu:0");
  model.to(device);
  if(board == std::nullopt) {
    board = Board(fen);
  } else {
    board.value().makeMove(uci::uciToMove(board.value(), move));
  }
  std::cout << board.value().at(Square(20)) << std::endl;
  //std::cout << board.value().getFen() << std::endl;
  int color = board.value().sideToMove() == Color::WHITE ? 1 : -1;

  std::pair<int, std::optional<Move>> minimax_result = minimax_internal(board.value(), depth, -100000000, 100000000, color);
  int best_score = minimax_result.first;
  if(minimax_result.second == std::nullopt) {
    std::cout << "game over" << std::endl;
    return "";
  }
  Move best_move = minimax_result.second.value();

  std::stringstream buffer;
  buffer << best_move << std::endl;
  std::string output = buffer.str();
  //std::cout << output << std::endl;

  board.value().makeMove(best_move);

  return output;
}

int main() {
  //minimax("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 4);
  //minimax("8/5p1p/4p1p1/p5P1/P5P1/2b3K1/8/1k3q2 w - - 42 67", 4);

}
