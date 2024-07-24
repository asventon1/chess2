# Chess
To set up the enviorment run
```
./pip_setup.sh
source chess2env/bin/activate
```
Then to build the cpp part go to the cpp directory and run
```
git clone https://github.com/Disservin/chess-library.git
./run.sh
```
Then to play against the AI go back to the main directory and run
```
python main.py
```
Then go to localhost:5000 in a web browser to play

To train a new model follow these steps

To download and extract the data run
```
wget https://database.lichess.org/lichess_db_eval.jsonl.zst
unzstd lichess_db_eval.jsonl.zst
```
Then to split the data into chunks run
```
./make_data_range.sh lichess_db_eval.jsonl 
```
Then to parse the data into an npy file you have to install rust <br>
then go to the rust directory and run
```
cargo run
```
Then to train run
```
python ai.py
```
