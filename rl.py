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
import logging
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)
board = chess.Board()
#model = ai.ModelSmall().to(ai.device)
model = torch.jit.load("stockfish_model_rl.pt").to(ai.device)
model.eval()            
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
ai.model_global = model

white_boards = torch.Tensor([ai.board_array_from_fen(board.fen())])
black_boards = torch.Tensor([ai.board_array_from_fen(board.fen())])
white_evals = torch.Tensor([0.5])
black_evals = torch.Tensor([0.5])

train_boards = torch.Tensor([ai.board_array_from_fen(board.fen())])
train_evals = torch.Tensor([0.5])

@app.route('/')
@app.route('/index')
def index():
    return flask.send_file("rl.html")

def run_rl():
    global board
    global white_boards
    global black_boards
    global white_evals
    global black_evals
    global train_boards
    global train_evals
    #time.sleep(1)
    best_move, new_boards, new_evals = pick_move_rl(board)
    if(board.turn == chess.WHITE):
        white_boards = torch.cat((white_boards, new_boards), 0)
        white_evals = torch.cat((white_evals, new_evals), 0)
    else:
        black_boards = torch.cat((black_boards, new_boards), 0)
        black_evals = torch.cat((black_evals, new_evals), 0)

    board.push(best_move)

    outcome = board.outcome()
    if outcome != None:
        winner = outcome.winner
        termination = outcome.termination
        if termination == chess.Termination.CHECKMATE:
            if winner == chess.WHITE:
                print("White wins")
                train_boards = torch.cat((train_boards, white_boards), 0)
                train_evals = torch.cat((train_evals, white_evals), 0)
                train_boards = torch.cat((train_boards, black_boards), 0)
                train_evals = torch.cat((train_evals, black_evals*-1+1), 0)
            else:
                print("Black wins")
                train_boards = torch.cat((train_boards, white_boards), 0)
                train_evals = torch.cat((train_evals, white_evals*-1+1), 0)
                train_boards = torch.cat((train_boards, black_boards), 0)
                train_evals = torch.cat((train_evals, black_evals), 0)
        else:
            print("It's a tie")

        print(train_evals)
        if(len(train_evals) > 100000):
            train_evals = train_evals[-100000:]

        train_dataset = TensorDataset(train_boards, train_evals.reshape((len(train_evals),1)))
        train_kwargs = {'batch_size': 64}
        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        if(len(train_evals) > 1):
            ai.train(100, model, ai.device, train_loader, optimizer, 0)
        board = chess.Board()
        white_boards = torch.Tensor([ai.board_array_from_fen(board.fen())])
        black_boards = torch.Tensor([ai.board_array_from_fen(board.fen())])
        white_evals = torch.Tensor([0.5])
        black_evals = torch.Tensor([0.5])
        torch.jit.script(model).save("stockfish_model_rl.pt")

@app.route('/board.svg/<bs>')
def boardRoute(bs):
    global board
    run_rl()

    boardImage = chess.svg.board(
        board
    )
    response = app.response_class(
        response=boardImage,
        status=200,
        mimetype='image/svg+xml; charset=utf-8'
    )
    return response

#app.run(host='0.0.0.0')
while True:
    run_rl()
