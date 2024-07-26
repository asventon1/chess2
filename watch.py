import chess
import chess.engine
import chess.pgn
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
import logging

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)
#board = chess.Board("3q2rk/p4p2/4p2p/3P2p1/1P2B3/5P1P/4K2r/3R4 w - - 14 40")
#board = chess.Board("8/5p1p/4p1p1/p5P1/P5P1/2b3K1/8/1k3q2 w - - 42 67")
board = chess.Board()
currentSquare = chess.A1
moves = []
ai.load_model()
engine = chess.engine.SimpleEngine.popen_uci(r"/usr/bin/stockfish")
engine.configure({"Skill Level": 8})



@app.route('/')
@app.route('/index')
def index():
    return flask.send_file("rl.html")

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
    
my_times = []
opp_times = []

depth = 3
@app.route('/board.svg/<bs>')
def boardRoute(bs):
    global board
    global moves
    global currentSquare
    global depth
    global engine
    #print(square, moves)
    #time.sleep(1)
    if(not board.is_game_over()):
        if(len(my_times) > 0 and len(opp_times) > 0):
            print("my average: {}   opp average: {}".format(sum(my_times) / len(my_times), sum(opp_times) / len(opp_times)))
        if(board.turn == chess.WHITE):
            time2 = time.time()
            best_move = engine.play(board, chess.engine.Limit(time=0.1)).move
            time3 = time.time()
            print("opps {}".format(time3-time2))
            opp_times.append(time3-time2)
            board.push(best_move)
        else:
            #best_move = pick_move(board)
            #time1 = time.time()
            #(best_score, best_move) = pick_move_minimax(board, depth, -10000000, 10000000, -1)
            time2 = time.time()
            #print("old version {}".format(time2-time1))
            depth=2
            best_move_str = cpp.minimax(board.fen(), board.peek().uci(), depth, 0.9)
            if(best_move_str != ""):
                best_move = chess.Move.from_uci(best_move_str.strip())
            time3 = time.time()
            print("me {}  depth {}".format(time3-time2, depth))
            my_times.append(time3-time2)
            if(time3-time2 > 200 and depth > 2):
                depth -= 1
            elif(time3-time2 < 2):
                depth += 1
            if(best_move_str != ""):
                board.push(best_move)
    else:
        game = chess.pgn.Game()
        game.headers["Event"] = "Example" 
        game.add_line(board.move_stack)
        print(game)
        time.sleep(100000)

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

