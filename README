# Chess
To set up the enviorment run
```
./pip_setup.sh
source env/bin/activate
```
Then go to this link https://database.lichess.org/#evals and download and extract the lichess stockfish dataset

Then to split the data into chunks run
```
./make_data_range.sh <whatever the datafile is called>
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
Then to play against the AI run
```
python main.py
```
Then go to localhost:5000 in a web browser to play
