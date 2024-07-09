import tensorflow as tf
import json
import numpy as np
import math

model_global = None

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64*13+1, activation='leaky_relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(64*13+1, activation='leaky_relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(400, activation='leaky_relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(400, activation='leaky_relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(200, activation='leaky_relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(200, activation='leaky_relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(100, activation='leaky_relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(100, activation='leaky_relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(1)
    ])
    loss_fn = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer='SGD',
                  loss=loss_fn)
    return model

def adjusted_sigmoid(x):
    return 1/(1+math.exp(-x/300))

def board_array_from_fen(fen_input):
    current_board = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for i in range(0, 64)])
    current_color = 0

    whole_fen = fen_input.split()
    fen = whole_fen[0]
    color = whole_fen[1]
    if color == 'w':
        current_color = 1
    square = 0
    for c in fen:
        if(c == '/'):
            pass
        elif(c.isalpha()):
            #print(current_board)
            #print(square)
            current_board[square][0] = 0
            if(c == 'p'):
                current_board[square][1] = 1
            elif(c == 'n'):
                current_board[square][2] = 1
            elif(c == 'b'):
                current_board[square][3] = 1
            elif(c == 'r'):
                current_board[square][4] = 1
            elif(c == 'q'):
                current_board[square][5] = 1
            elif(c == 'k'):
                current_board[square][6] = 1
            elif(c == 'P'):
                current_board[square][7] = 1
            elif(c == 'N'):
                current_board[square][8] = 1
            elif(c == 'B'):
                current_board[square][9] = 1
            elif(c == 'R'):
                current_board[square][10] = 1
            elif(c == 'Q'):
                current_board[square][11] = 1
            elif(c == 'K'):
                current_board[square][12] = 1
            square += 1
        else:
            square += int(c)

    current_board = np.append(current_board, current_color)
    return current_board

'''
format is 
none    pawn  knight  bishop  rook  queen   king                                                        whose turn (1 for white)       evaluation for white
[[1,     0,      0,     0,      0,     0,     0] * 2 for the white pieces * 64 for each square on the board,       0,                            0]
'''
def board_array_from_json(j):
    board_array = None
    eval_array = None
    for i in range(len(j)):
        if(i % int(len(j)/1000) == 0):
            print(str(i/len(j)*100)+"% done")

        current_eval = 0
        eval = j[i]["evals"][0]["pvs"][0]
        if("cp" in eval):
            current_eval = adjusted_sigmoid(eval["cp"])
        else:
            continue
        if(type(eval_array) == type(None)):
            eval_array = np.array([current_eval])
        else:
            eval_array = np.append(eval_array, [current_eval])

        current_board = board_array_from_fen(j[i]["fen"])
        
        #print(board_array)
        if(type(board_array) == type(None)):
            board_array = np.array([current_board])
        else:
            board_array = np.append(board_array, [current_board], axis=0)
    return (board_array, eval_array)

def fen_from_board_array(arr):
    board = arr[:-1]
    board = np.reshape(board, (64, 13))
    line_count = 0
    empty_count = 0
    fen = ""
    for v in board:
        if(line_count == 8):
            if(empty_count != 0):
                fen += str(empty_count)
                empty_count = 0
            fen += "/"
            line_count = 0
        line_count += 1
        if(v[0] == 1):
            empty_count += 1
        else:
            if(empty_count != 0):
                fen += str(empty_count)
                empty_count = 0
            if(v[1] == 1):
                fen += "p"
            elif(v[2] == 1):
                fen += "n"
            elif(v[3] == 1):
                fen += "b"
            elif(v[4] == 1):
                fen += "r"
            elif(v[5] == 1):
                fen += "q"
            elif(v[6] == 1):
                fen += "k"
            elif(v[7] == 1):
                fen += "P"
            elif(v[8] == 1):
                fen += "N"
            elif(v[9] == 1):
                fen += "B"
            elif(v[10] == 1):
                fen += "R"
            elif(v[11] == 1):
                fen += "Q"
            elif(v[12] == 1):
                fen += "K"
    if(empty_count != 0):
        fen += str(empty_count)
        empty_count = 0
    fen += " "
    if(arr[-1] == 0):
        fen += "b"
    else:
        fen += "w"

    fen += " - - 0 1"
    return fen
     
def save_data_as_numpy():
    f = open("start_data_mil.json")

    data = json.load(f)

    boards, evals = board_array_from_json(data)
    with open('stockfish_data.npy', 'wb') as of:
        np.save(of, boards)
        np.save(of, evals)
    #board_fen = fen_from_board_array(boards[2])

def load_data_from_numpy(start):
    with open('rust/array.npy', 'rb') as of:
        for i in range(start):
            new_boards = np.load(of)
            new_evals = np.load(of)
        boards = np.load(of)
        evals = np.load(of)
        for i in range(9):
            new_boards = np.load(of)
            new_evals = np.load(of)
            boards = np.concatenate((boards, new_boards), axis=0)
            evals = np.concatenate((evals, new_evals), axis=0)
        return (boards, evals)

def train_model():
    model = create_model()
    #model = tf.keras.models.load_model('stockfish_model2.keras')
    while True:
        for i in range(4):
            boards, evals = load_data_from_numpy(i*10)
            test_train_split = 5.0/6.0
            boards_train = boards[:int(len(boards)*test_train_split)]
            evals_train = evals[:int(len(evals)*test_train_split)]
            boards_test = boards[int(len(boards)*test_train_split):]
            evals_test = evals[int(len(evals)*test_train_split):]
            loss_fn = tf.keras.losses.MeanSquaredError()

            print(boards_train.shape, boards_test.shape)
            print(tf.keras.ops.average(loss_fn.call(evals_test, model(boards_test))))
            model.fit(boards_train, evals_train, epochs=1)
            print(tf.keras.ops.average(loss_fn.call(evals_test, model(boards_test))))
            #print(tf.keras.ops.average(loss_fn.call(evals_train, model(boards_train))))
            #print(model(boards).numpy())
            model.save("stockfish_model2.keras")

def load_model():
    global model_global
    model_global = tf.keras.models.load_model('stockfish_model2.keras')

def use_model(input_board):
    global model_global
    return model_global(np.array([input_board]))


if __name__ == "__main__":
    #save_data_as_numpy()
    train_model()
else:
    load_model()
