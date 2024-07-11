import chess
import chess.svg
import flask
from flask import Flask
import random
import ai

board = chess.Board()
currentSquare = chess.A1
moves = []

app = Flask(__name__)
@app.route('/')
@app.route('/index')
def index():
    return flask.send_file("test.html")

@app.route('/outcome')
def outcomeRoute():
    text = "Game is still playing"
    outcome = board.outcome()
    if outcome != None:
        winner = outcome.winner
        termination = outcome.termination
        if termination == chess.Termination.CHECKMATE:
            if winner == chess.WHITE:
                text = "White wins"
            else:
                text = "Black wins"
        else:
            text = "It's a tie"
    response = app.response_class(
        response=text,
        status=200,
        mimetype='text/plain'
    )
    return response
    
def pick_move(board):
    best_score = 100
    best_move = None
    for m in board.legal_moves: 
        board.push(m)
        board_array = ai.board_array_from_fen(board.fen())
        score = ai.use_model(board_array)
        if(score < best_score):
            best_score = score
            best_move = m
        board.pop()
    return best_move

def pick_move_minimax(board, depth, color):
    if(len(list(board.legal_moves)) == 0 or depth == 0):
        board_array = ai.board_array_from_fen(board.fen())
        return (color * ai.use_model(board_array), None)
        #return (1, None)
    best_score = -1000000
    best_move = None
    for m in board.legal_moves: 
        board.push(m)
        (score, move) = pick_move_minimax(board, depth-1, -color)
        score = -score
        if(score > best_score):
            best_score = score
            best_move = m
        board.pop()
    return (best_score, best_move)

@app.route('/board.svg/<xPos>/<yPos>/<bs>')
def boardRoute(xPos, yPos, bs):
    global board
    global moves
    global currentSquare
    square = chess.square(int(xPos), int(yPos)) 
    #print(square, moves)
    if(square in moves):
        board.push(board.find_move(currentSquare, square))
        moves = []
        #best_move = pick_move(board)
        (best_score, best_move) = pick_move_minimax(board, 3, -1)
        if(best_move != None):
            board.push(best_move)
    else:
        moves = [x.to_square for x in board.legal_moves if x.from_square == square]
    currentSquare = square

    #print(moves)

    boardImage = chess.svg.board(
        board,
        fill=dict.fromkeys(moves, "#cc0000cc"),
    )

    #print(boardImage)

    response = app.response_class(
        response=boardImage,
        status=200,
        mimetype='image/svg+xml; charset=utf-8'
    )
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0')


