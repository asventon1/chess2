import chess
import time
import chess.svg
import flask
from flask import Flask
import random
import ai
import numpy as np
import torch
from player import *
import cpp_part as cpp

app = Flask(__name__)
#board = chess.Board("3q2rk/p4p2/4p2p/3P2p1/1P2B3/5P1P/4K2r/3R4 w - - 14 40")
#board = chess.Board("8/5p1p/4p1p1/p5P1/P5P1/2b3K1/8/1k3q2 w - - 42 67")
#board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1")
#board = chess.Board("6k1/1b3pp1/2p2n1p/2PpqP2/rp2p3/1NP1P3/p2Q1RPP/R5K1 w - - 0 27")
#board = chess.Board("4brk1/N1nn1p1p/1p1qp1p1/3p4/3P4/1Q1BPN1P/2P2PP1/1R4K1 w - - 0 28")
#board = chess.Board("r1bqk2r/ppp2ppp/2n1p3/6N1/Qbp5/6P1/PP1PPP1P/RNB1K2R w KQkq - 0 8")
board = chess.Board()
currentSquare = chess.A1
moves = []
#ai.load_model()


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
    


depth = 2
@app.route('/board.svg/<xPos>/<yPos>/<bs>')
def boardRoute(xPos, yPos, bs):
    global board
    global moves
    global currentSquare
    global depth
    square = chess.square(int(xPos), int(yPos)) 
    #print(square, moves)
    if(square in moves):
        board.push(board.find_move(currentSquare, square))
        moves = []
        #best_move = pick_move(board)
        #time1 = time.time()
        #(best_score, best_move) = pick_move_minimax(board, depth, -10000000, 10000000, -1)
        time2 = time.time()
        #print("old version {}".format(time2-time1))
        best_move_str = cpp.minimax(board.fen(), board.peek().uci(), depth)
        if(best_move_str != ""):
            best_move = chess.Move.from_uci(best_move_str.strip())
        else:
            game = chess.pgn.Game()
            game.headers["Event"] = "Example" 
            game.add_line(board.move_stack)
            print(game)
            time.sleep(100000)
        time3 = time.time()
        print("new version {}  depth {}".format(time3-time2, depth))
        if(time3-time2 > 10 and depth > 2):
            depth -= 1
        elif(time3-time2 < 0.1):
            depth += 1
        if(best_move != None):
            board.push(best_move)
        print(board.fen())
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

app.run(host='0.0.0.0')

