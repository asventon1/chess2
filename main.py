import chess
import time
import chess.svg
import flask
from flask import Flask
import random
import ai
import numpy as np
import torch

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


def fill(board, depth, color):
    #if(type(boards) == type(None)):
    #    print("none")
    #else:
    #    print(boards.shape)
    boards = None
    if(len(list(board.legal_moves)) == 0 or depth == 0):
        board_array = ai.board_array_from_fen(board.fen())
        #print(board_array)
        if(type(boards) == type(None)):
            boards = np.array([board_array])
        else:
            boards = np.append(boards, np.array([board_array]), axis=0)
        return boards
        #return (1, None)
    for m in board.legal_moves: 
        board.push(m)
        new_boards = fill(board, depth-1, -color)
        if(type(boards) == type(None)):
            boards = new_boards
        else:
            boards = np.append(boards, new_boards, axis=0)
        board.pop()
    return boards

def run(board, depth, color, evals, index):
    if(len(list(board.legal_moves)) == 0 or depth == 0):
        board_array = ai.board_array_from_fen(board.fen())
        return (color * evals[index], None, index+1)
        #return (1, None)
    best_score = -1000000
    best_move = None
    for m in board.legal_moves: 
        board.push(m)
        (score, move, new_index) = run(board, depth-1, -color, evals, index)
        index = new_index
        score = -score
        if(score > best_score):
            best_score = score
            best_move = m
        board.pop()
    return (best_score, best_move, index)

def pick_move_minimax2(board_o, depth_o, color_o):
    boards = fill(board_o, depth_o, color_o)
    evals = ai.use_model(boards)
    evals = torch.flatten(evals)
    #print(boards.shape)
    #print(evals.shape)
    (score, move, index) = run(board_o, depth_o, color_o, evals, 0)
    return (score, move)

def pick_move_minimax(board, depth, alpha, beta, color):
    outcome = board.outcome()
    if outcome != None: 
        winner = outcome.winner
        termination = outcome.termination
        if termination == chess.Termination.CHECKMATE:
            if winner == chess.WHITE:
                return (1000, None)
            else:
                return (-1000, None)
        else:
            return (0.5, None)
    if(len(list(board.legal_moves)) == 0 or depth == 0):
        board_array = ai.board_array_from_fen(board.fen())
        return (color * ai.use_model(board_array), None)
        #return (1, None)
    best_score = -1000000
    best_move = None
    for m in board.legal_moves: 
        board.push(m)
        (score, move) = pick_move_minimax(board, depth-1, -beta, -alpha, -color)
        score = -score
        if(score > best_score):
            best_score = score
            best_move = m
        if(best_score > alpha):
            alpha = best_score
        board.pop()
        if(alpha >= beta):
            break
    return (best_score, best_move)

class ttTableEntry:
    value = None
    flag = None
    depth = None
ttTable = {}

def pick_move_minimax_table(board, depth, alpha, beta, color):
    global ttTable
    alphaOrig = alpha
    fen = board.fen()
    if(fen in ttTable):
        entry = ttTable[fen]
        if(entry.depth >= depth):
            if(entry.flag == "EXACT"):
                return entry.value
            elif(entry.flag == "LOWERBOUND" and entry.value > alpha):
                alpha = entry.value 
            elif(entry.flag == "UPERBOUND" and entry.value < beta):
                beta = entry.value
            if(alpha > beta):
                return entry.value

    outcome = board.outcome()
    if outcome != None: 
        winner = outcome.winner
        termination = outcome.termination
        if termination == chess.Termination.CHECKMATE:
            if winner == chess.WHITE:
                return (1000, None)
            else:
                return (-1000, None)
        else:
            return (0.5, None)
    if(len(list(board.legal_moves)) == 0 or depth == 0):
        board_array = ai.board_array_from_fen(board.fen())
        return (color * ai.use_model(board_array), None)
        #return (1, None)
    best_score = -1000000
    best_move = None
    for m in board.legal_moves: 
        board.push(m)
        (score, move) = pick_move_minimax(board, depth-1, -beta, -alpha, -color)
        score = -score
        if(score > best_score):
            best_score = score
            best_move = m
        if(best_score > alpha):
            alpha = best_score
        board.pop()
        if(alpha >= beta):
            break
    entry = ttTableEntry()
    entry.value = best_score
    if(best_score <= alphaOrig):
        entry.flag = "UPERBOUND"
    elif(best_score >= beta):
        entry.flag = "LOWERBOUND"
    else:
        entry.flag = "EXACT"
    entry.depth = depth
    ttTable[fen] = entry
    return (best_score, best_move)

def pick_move_network(board):
    board_array = ai.board_array_from_fen(board.fen())


depth = 3
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
        depth=1
        (best_score, best_move) = pick_move_minimax(board, depth, -10000000, 10000000, -1)
        time3 = time.time()
        print("new version {}".format(time3-time2))
        if(time3-time2 > 1):
            depth -= 1
        elif(time3-time2 < 0.1):
            depth += 1
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


