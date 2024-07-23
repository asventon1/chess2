import chess
import sys
import concurrent.futures
import copy
import threading
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

executor = concurrent.futures.ThreadPoolExecutor(max_workers=24)

# Genetic algorithm parameters
population_size = 10
mutation_rate = 0.1

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)
board = chess.Board()
#model = ai.ModelSmall().to(ai.device)
model = torch.jit.load("stockfish_model_rl.pt").to(ai.device)
model.eval()            
print(model)
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

def mutate(model):
    for param in model.parameters():
        if torch.rand(1).item() < mutation_rate:
            param.data += torch.randn_like(param.data) * 0.1  # Adding Gaussian noise with std=0.1
    return model

def play_game(board, model1, model2, i, j, global_board = True):
    #global board
    while True:
        #time.sleep(0.2)
        current_model = model1 if(board.turn == chess.WHITE) else model2
        best_move = None
        if(global_board):
            new_board = board.copy()
            best_move = pick_move_model(new_board, current_model)
        else:
            best_move = pick_move_model(board, current_model)
        board.push(best_move)
        outcome = board.outcome()
        if outcome != None:
            winner = outcome.winner
            termination = outcome.termination
            if termination == chess.Termination.CHECKMATE:
                if winner == chess.WHITE:
                    print("White wins")
                    return (1, 0, i, j)
                else:
                    print("Black wins")
                    return (0, 1, i, j)
            else:
                #print("It's a tie")
                return (0.5, 0.5, i, j)

def make_new_models(winner):
    return [mutate(copy.deepcopy(winner)) for _ in range(population_size)]

def run_rl_genetic(multithread = False):
    global board
    starting_model = torch.jit.load("stockfish_model_rl_genetic.pt").to(ai.device)
    starting_model.eval()
    #models = [ai.ModelSmall().to(ai.device) for _ in range(population_size)] 
    models = make_new_models(starting_model)
    for _ in range(10000000):
        for model in models:
            model.eval()
        scores = torch.zeros(population_size)
        futures = []
        for i in range(population_size):
            for j in range(i+1, population_size):
                if(multithread):
                    if(i == 0 and j == 1):
                        board = chess.Board()
                        futures.append(executor.submit(play_game, board, models[i], models[j], i, j))
                    else:
                        futures.append(executor.submit(play_game, chess.Board(), models[i], models[j], i, j, False))

                else:
                    board = chess.Board()
                    score1, score2, i, j = play_game(board, models[i], models[j], i, j)
                    scores[i] += score1
                    scores[j] += score2
        if(multithread):
            for v in futures:
                (score1, score2, i, j) = v.result()
                scores[i] += score1
                scores[j] += score2
        winner = torch.argmax(scores)
        print(scores)
        torch.jit.script(model).save("stockfish_model_rl_genetic.pt")
        models = make_new_models(models[winner])

def run_rl():
    global board
    global white_boards
    global black_boards
    global white_evals
    global black_evals
    global train_boards
    global train_evals
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
        model.train()
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
            train_boards = train_boards[-100000:]

        train_dataset = TensorDataset(train_boards, train_evals.reshape((len(train_evals),1)))
        train_kwargs = {'batch_size': 64}
        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        if(len(train_evals) > 1):
            ai.train(1500, model, ai.device, train_loader, optimizer, 0)
            ai.train(1500, model, ai.device, train_loader, optimizer, 0)
        board = chess.Board()
        white_boards = torch.Tensor([ai.board_array_from_fen(board.fen())])
        black_boards = torch.Tensor([ai.board_array_from_fen(board.fen())])
        white_evals = torch.Tensor([0.5])
        black_evals = torch.Tensor([0.5])
        torch.jit.script(model).save("stockfish_model_rl.pt")
        model.eval()

@app.route('/board.svg/<bs>')
def boardRoute(bs):
    global board
    #run_rl()

    boardImage = chess.svg.board(
        board
    )
    response = app.response_class(
        response=boardImage,
        status=200,
        mimetype='image/svg+xml; charset=utf-8'
    )
    return response

def run_flask():
    app.run(host='0.0.0.0')
def run_learning():
    while True:
        run_rl()

if(sys.argv[1] == "single"):
    t1 = threading.Thread(target=run_flask)
    t2 = threading.Thread(target=run_learning)

    t1.start()
    t2.start()
    t1.join()
    t2.join()
else:
    #t1 = threading.Thread(target=run_flask)
    t2 = threading.Thread(target=run_rl_genetic, kwargs={"multithread": True})

    #t1.start()
    t2.start()
    #t1.join()
    t2.join()
    executor.shutdown()
