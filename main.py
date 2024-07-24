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
board = chess.Board()
currentSquare = chess.A1
moves = []
ai.load_model()


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
    


depth = 4
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
        best_move_str = cpp.minimax(board.fen(), depth)
        best_move = chess.Move.from_uci(best_move_str.strip())
        time3 = time.time()
        print("new version {}".format(time3-time2))
        if(time3-time2 > 3):
            depth -= 1
        elif(time3-time2 < 0.5):
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

