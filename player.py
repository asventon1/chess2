import chess
import time
import chess.svg
import flask
from flask import Flask
import random
import ai
import numpy as np
import torch

def pick_move_first(board):
    return random.choice(list(board.legal_moves))

def pick_move_rl(board):
    moves = list(board.legal_moves)
    boards = []
    evals = []
    for m in moves: 
        board.push(m)
        board_array = ai.board_array_from_fen(board.fen())
        boards.append(board_array)
        score = ai.use_model(board_array).item() + random.random()/100
        outcome = board.outcome()
        if outcome != None: 
            winner = outcome.winner
            termination = outcome.termination
            if termination == chess.Termination.CHECKMATE:
                if winner == chess.WHITE:
                    score = 1
                else:
                    score = 0
            else:
                score = 0.5
        evals.append(score)
        board.pop()
    #best_move = random.choices(moves, weights=torch.nn.functional.relu(torch.Tensor(evals)))
    if(board.turn == chess.WHITE):
        best_move = moves[torch.argmax(torch.Tensor(evals))]
    else:
        best_move = moves[torch.argmin(torch.Tensor(evals))]

    return (best_move, torch.Tensor(boards), torch.Tensor(evals))

def pick_move_model(board, model):
    best_score = 10000 if board.turn == chess.BLACK else -10000
    best_move = None
    for m in board.legal_moves: 
        board.push(m)
        board_array = ai.board_array_from_fen(board.fen())
        outcome = board.outcome()
        score = 0
        if outcome != None: 
            winner = outcome.winner
            termination = outcome.termination
            if termination == chess.Termination.CHECKMATE:
                if winner == chess.WHITE:
                    score = 1000
                else:
                    score = -1000
            else:
                score = 0.5
        else:
            score = model(torch.Tensor(np.array([board_array])).to(ai.device))
        board.pop()
        if((score < best_score and board.turn == chess.BLACK) or (score > best_score and board.turn == chess.WHITE)):
            best_score = score
            best_move = m
    return best_move

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
            return (-1, None)
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
