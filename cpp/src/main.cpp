#include <torch/torch.h>
#include <torch/script.h>
#include "main.h"
#include "chess.hpp"
#include <string>

using namespace chess;

std::vector<std::string> get_words(std::string s){
    std::vector<std::string> res;
    int pos = 0;
    while(pos < s.size()){
        pos = s.find(" ");
        res.push_back(s.substr(0,pos));
        s.erase(0,pos+1);
   }
    return res;
}

std::array<char, 64*13+1> board_array_from_fen(std::string fen_input) {
  std::array<char, 64*13+1> current_board;
  for(unsigned int i = 0; i < 64; i++) {
    current_board[i*13] = 1;
    for(unsigned int j = 0; j < 12; j++){
      current_board[i*12+j+1] = 0;
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
    } else if(isalpha('w')) {
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

double get_board_value(Board board) {
  auto model = torch::jit::load("/home/adam/stuff/python/chess2/stockfish_model_small.pt");
  //model.eval();
  auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA, 1);
  std::array<char, 833> array_data = board_array_from_fen(board.getFen());

  //torch::Tensor prediction = model.forward(torch::from_blob(my_input, {833}, options));
  //std::cout << prediction << std::endl;
  return 1;
}

std::pair<int, std::optional<Move>> minimax_internal(Board board, int depth, double alpha, double beta, int color) {
  GameResult outcome = board.isGameOver().second;
  switch(outcome) {
    case GameResult::NONE:
      break;
    case GameResult::WIN:
    case GameResult::LOSE:
      if(board.sideToMove() == Color::WHITE) return std::make_pair(1000, std::nullopt);
      else return std::make_pair(-1000, std::nullopt);
      break;
    case GameResult::DRAW:
      return std::make_pair(-1, std::nullopt);
  }
  if(depth == 0) {
    return std::make_pair(color*get_board_value(board), std::nullopt);
  }
  int best_score = -10000000;
  std::optional<Move> best_move = std::nullopt;
  Movelist moves;
  movegen::legalmoves(moves, board);
  for(const auto &m : moves) {
    board.makeMove(m);
    std::pair<int, std::optional<Move>> minimax_result = minimax_internal(board, depth-1, -beta, -alpha, -color);
    int score = minimax_result.first;
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
    if(alpha > beta) {
      break;
    }
  }
  return std::make_pair(best_score, best_move);
}

std::string minimax(const std::string &fen){
  Board board = Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

  std::pair<int, std::optional<Move>> minimax_result = minimax_internal(board, 2, -100000000, 100000000, -1);
  int best_score = minimax_result.first;
  Move best_move = minimax_result.second.value();
  std::cout << best_move << std::endl;

  return fen + " world";
}