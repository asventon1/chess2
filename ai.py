#import tensorflow as tf
import time
import json
import numpy as np
import math
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

model_global = None

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


class MoveModel1(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(64*13+1, 800), nn.LeakyReLU(), #nn.Dropout(p=0.1),
            nn.Linear(800, 800), nn.LeakyReLU(), #nn.Dropout(p=0.1),
            nn.Linear(800, 800), nn.LeakyReLU(), #nn.Dropout(p=0.1),
            nn.Linear(800, 800), nn.LeakyReLU(), #nn.Dropout(p=0.1),
            nn.Linear(800, 400), nn.LeakyReLU(), #nn.Dropout(p=0.1),
            nn.Linear(400, 200), nn.LeakyReLU(), #nn.Dropout(p=0.1),
            nn.Linear(200, 100), nn.LeakyReLU(), #nn.Dropout(p=0.1),
            nn.Linear(100, 64),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

class MoveModel2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(64*13+1+64, 800), nn.LeakyReLU(), #nn.Dropout(p=0.1),
            nn.Linear(800, 800), nn.LeakyReLU(), #nn.Dropout(p=0.1),
            nn.Linear(800, 800), nn.LeakyReLU(), #nn.Dropout(p=0.1),
            nn.Linear(800, 800), nn.LeakyReLU(), #nn.Dropout(p=0.1),
            nn.Linear(800, 400), nn.LeakyReLU(), #nn.Dropout(p=0.1),
            nn.Linear(400, 200), nn.LeakyReLU(), #nn.Dropout(p=0.1),
            nn.Linear(200, 100), nn.LeakyReLU(), #nn.Dropout(p=0.1),
            nn.Linear(100, 64),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

class ModelSmall(nn.Module):

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            #torch.nn.init.xavier_uniform_(m.weight)
            m.weight.data.fill_(1)
            m.bias.data.fill_(0)

    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(64*13+1, 100), nn.LeakyReLU(), # nn.Dropout(p=0.1), nn.BatchNorm1d(800),
            nn.Linear(100, 50), nn.LeakyReLU(), # nn.Dropout(p=0.1), nn.BatchNorm1d(100),
            nn.Linear(50, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

class Model(nn.Module):

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            #torch.nn.init.xavier_uniform_(m.weight)
            m.weight.data.fill_(1)
            m.bias.data.fill_(0)

    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(64*13+1, 800), nn.LeakyReLU(), nn.Dropout(p=0.1), # nn.BatchNorm1d(800),
            nn.Linear(800, 800), nn.LeakyReLU(), nn.Dropout(p=0.1), # nn.BatchNorm1d(800),
            nn.Linear(800, 800), nn.LeakyReLU(), nn.Dropout(p=0.1), # nn.BatchNorm1d(800),
            nn.Linear(800, 800), nn.LeakyReLU(), nn.Dropout(p=0.1), # nn.BatchNorm1d(800),
            nn.Linear(800, 400), nn.LeakyReLU(), nn.Dropout(p=0.1), # nn.BatchNorm1d(400),
            nn.Linear(400, 400), nn.LeakyReLU(), nn.Dropout(p=0.1), # nn.BatchNorm1d(400),
            nn.Linear(400, 400), nn.LeakyReLU(), nn.Dropout(p=0.1), # nn.BatchNorm1d(400),
            nn.Linear(400, 400), nn.LeakyReLU(), nn.Dropout(p=0.1), # nn.BatchNorm1d(400),
            nn.Linear(400, 200), nn.LeakyReLU(), nn.Dropout(p=0.1), # nn.BatchNorm1d(200),
            nn.Linear(200, 200), nn.LeakyReLU(), nn.Dropout(p=0.1), # nn.BatchNorm1d(200),
            nn.Linear(200, 200), nn.LeakyReLU(), nn.Dropout(p=0.1), # nn.BatchNorm1d(200),
            nn.Linear(200, 200), nn.LeakyReLU(), nn.Dropout(p=0.1), # nn.BatchNorm1d(200),
            nn.Linear(200, 100), nn.LeakyReLU(), nn.Dropout(p=0.1), # nn.BatchNorm1d(100),
            nn.Linear(100, 100), nn.LeakyReLU(), nn.Dropout(p=0.1), # nn.BatchNorm1d(100),
            nn.Linear(100, 100), nn.LeakyReLU(), nn.Dropout(p=0.1), # nn.BatchNorm1d(100),
            nn.Linear(100, 100), nn.LeakyReLU(), nn.Dropout(p=0.1), # nn.BatchNorm1d(100),
            nn.Linear(100, 1),
            nn.Sigmoid(),
        )
        self.linear_relu_stack2 = nn.Sequential(
        )

        #self.linear_relu_stack.apply(self.init_weights)
        #self.linear_relu_stack2.apply(self.init_weights)

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        #print(logits)
        #logits = self.linear_relu_stack2(logits)
        return logits


def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64*13+1, activation='sigmoid'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(400, activation='sigmoid'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(200, activation='sigmoid'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(100, activation='sigmoid'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(1)
    ])
    loss_fn = tf.keras.losses.MeanSquaredError()
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.01, decay_steps=20000, decay_rate=0.99)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule) 
    model.compile(optimizer="Adam",
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
    with open('rust/array2.npy', 'rb') as of:
        for i in range(start):
            new_boards = np.load(of)
            new_evals = np.load(of)
        boards = np.load(of)
        evals = np.load(of)
        for i in range(1):
            new_boards = np.load(of)
            new_evals = np.load(of)
            boards = np.concatenate((boards, new_boards), axis=0)
            evals = np.concatenate((evals, new_evals), axis=0)
        return (boards, evals)

def train(log_interval, model, device, train_loader, optimizer, epoch, reverse=False):
    model.train()
    average_loss = 0
    loss_len = 0
    losses = []
    last_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        #loss_fn = nn.CrossEntropyLoss()
        loss_fn = nn.MSELoss()
        loss = loss_fn(output, target)
        #print(list(zip(output.tolist(), target.tolist())))
        losses.append(loss.item())
        loss.backward()
        if(reverse):
            for p in model.parameters():
                p.grad *= -1
        #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        #print(optimizer)
        optimizer.step()
        average_loss += loss.item()
        loss_len += 1
    #if batch_idx % log_interval == 0:
    average_loss /= loss_len
    current_time = time.time()
    elapsed_time = current_time - last_time
    last_time = current_time
    #print(losses)
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime: {:.2f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), average_loss, elapsed_time))
    average_loss = 0
    loss_len = 0


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss_fn = nn.CrossEntropyLoss(reduction='sum')
            loss_fn = nn.MSELoss(reduction='sum')
            test_loss += loss_fn(output, target).item()  # sum up batch loss
            #pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            #correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}\n'.format(
        test_loss))
    return test_loss

def train_move_model():  
    epoch = 0

    model1 = MoveModel1().to(device)
    model2 = MoveModel2().to(device)
    #model = torch.jit.load("stockfish_model.pt").to(device)
    #model.eval()            
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.001)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
    with open("current_dataset.txt", "r") as f:
        current_dataset = int(f.read())
    while True:
        epoch += 1
        boards, evals = load_data_from_numpy(current_dataset)
        evals = np.transpose(evals, axes=(1,0,2))
        boards_evals = np.column_stack((boards, evals[0]))
        print(evals.shape)
        print(boards_evals.shape)
        test_train_split = 4/5
        boards_train = torch.Tensor(boards[:int(len(boards)*test_train_split)]) 
        boards_evals_train = torch.Tensor(boards_evals[:int(len(boards_evals)*test_train_split)]) 
        evals1_train = torch.Tensor(evals[0][:int(len(evals[0])*test_train_split)])
        evals2_train = torch.Tensor(evals[1][:int(len(evals[1])*test_train_split)])
        boards_test = torch.Tensor(boards[int(len(boards)*test_train_split):])
        evals1_test = torch.Tensor(evals[0][int(len(evals[0])*test_train_split):])
        evals2_test = torch.Tensor(evals[1][int(len(evals[1])*test_train_split):])
        boards_evals_test = torch.Tensor(boards_evals[int(len(boards_evals)*test_train_split):]) 

        print(boards_train.shape, boards_test.shape)

        train1_dataset = TensorDataset(boards_train, evals1_train)
        test1_dataset = TensorDataset(boards_test, evals1_test)
        train2_dataset = TensorDataset(boards_evals_train, evals2_train)
        test2_dataset = TensorDataset(boards_evals_test, evals2_test)
        train_kwargs = {'batch_size': 64}
        test_kwargs = {'batch_size': 64}
        train1_loader = torch.utils.data.DataLoader(train1_dataset,**train_kwargs)
        test1_loader = torch.utils.data.DataLoader(test1_dataset, **test_kwargs)
        train2_loader = torch.utils.data.DataLoader(train2_dataset,**train_kwargs)
        test2_loader = torch.utils.data.DataLoader(test2_dataset, **test_kwargs)


        train(10000, model1, device, train1_loader, optimizer1, epoch)
        train(10000, model2, device, train2_loader, optimizer2, epoch)
        test(model1, device, test1_loader) 
        test(model2, device, test2_loader) 

        #torch.save(model, "stockfish_model.pt")
        torch.jit.script(model1).save("stockfish_move_model1.pt")
        torch.jit.script(model2).save("stockfish_move_model2.pt")

        current_dataset += 1
        if(current_dataset >= 40):
            current_dataset = 0
        with open("current_dataset.txt", "w") as f:
            f.write(str(current_dataset))

def train_model():
    #model = create_model()
    #loss_fn = tf.keras.losses.MeanSquaredError()
    #model = tf.keras.models.load_model('stockfish_model2.keras')
    #print(model.optimizer.learning_rate)
    #lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.002, decay_steps=100000, decay_rate=0.99)
    #optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule) 
    #model.compile(optimizer=optimizer, loss=loss_fn)
    
    epoch = 0

    model = ModelSmall().to(device)
    #model = torch.jit.load("stockfish_model_small.pt").to(device)
    #model.eval()            
    #model.forward = model2.forward
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    with open("current_dataset.txt", "r") as f:
        current_dataset = int(f.read())

    test_loss = -100000
    while True:
        epoch += 1
        boards, evals = load_data_from_numpy(current_dataset*2)
        test_train_split = 49/50
        boards_train = torch.Tensor(boards[:int(len(boards)*test_train_split)])
        evals_train = torch.Tensor(evals.reshape((len(evals),1))[:int(len(evals)*test_train_split)])
        boards_test = torch.Tensor(boards[int(len(boards)*test_train_split):])
        evals_test = torch.Tensor(evals.reshape((len(evals),1))[int(len(evals)*test_train_split):])

        print(boards_train.shape, boards_test.shape)

        train_dataset = TensorDataset(boards_train, evals_train)
        test_dataset = TensorDataset(boards_test, evals_test)
        train_kwargs = {'batch_size': 64}
        test_kwargs = {'batch_size': 64}
        train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)


        test(model, device, test_loader) 
        train(10000, model, device, train_loader, optimizer, epoch)
        test_loss += test(model, device, test_loader) 

        #torch.save(model, "stockfish_model.pt")
        torch.jit.script(model).save("stockfish_model_small.pt")

        current_dataset += 1
        if(current_dataset >= 20):
            print("wqpgohqwpoghqwpogihqrpogihqrpgoihqerpogihqrpogih" + str(test_loss))
            if test_loss > 0:
                with open("test_loss.txt", "a") as f:
                    f.write(str(test_loss/20) + "\n")
            test_loss = 0
            current_dataset = 0
        with open("current_dataset.txt", "w") as f:
            f.write(str(current_dataset))

        '''
        with tf.device('/CPU:0'):
            print(tf.keras.ops.average(loss_fn.call(evals_test, model(boards_test))))
        model.fit(boards_test, evals_test, epochs=1, batch_size=1)
        with tf.device('/CPU:0'):
            print(tf.keras.ops.average(loss_fn.call(evals_test, model(boards_test))))
        #print(tf.keras.ops.average(loss_fn.call(evals_train, model(boards_train))))
        #print(model(boards).numpy())
        model.save("stockfish_model2.keras")
        '''

def load_model():
    global model_global
    #model_global = tf.keras.models.load_model('stockfish_model2.keras')
    model_global = torch.jit.load("stockfish_model_small.pt").to(device)
    model_global.eval()

@torch.no_grad()
def use_model(input_board):
    global model_global
    return model_global(torch.Tensor(np.array([input_board])).to(device))


if __name__ == "__main__":
    #save_data_as_numpy()
    train_model()
