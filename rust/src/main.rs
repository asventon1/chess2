use std::fs;
use serde_json::Value;
use npyz::WriterBuilder;
use std::io::Write;

#[derive(Copy, Clone)]
enum DataType {
    Evals,
    Moves,
}

fn adjusted_sigmoid(x: f64) -> f64{
    1.0/(1.0+(-x/300.0).exp())
}

fn board_array_from_fen(fen_input: &str) -> Vec<u8> {
    let mut current_board = vec![[1,0,0,0,0,0,0,0,0,0,0,0,0]; 64];
    let whole_fen = fen_input.split_whitespace().collect::<Vec<&str>>();
    let fen = whole_fen[0];
    let color = whole_fen[1];
    let current_color = if color == "w" { 1 } else { 0 };
    let mut square = 0;
    for c in fen.chars(){
        if c == '/' {
        } else if c.is_alphabetic() {
            current_board[square][0] = 0;
            if c == 'p' { current_board[square][1] = 1 }
            else if c == 'n' { current_board[square][2] = 1 }
            else if c == 'b' { current_board[square][3] = 1 }
            else if c == 'r' { current_board[square][4] = 1 }
            else if c == 'q' { current_board[square][5] = 1 }
            else if c == 'k' { current_board[square][6] = 1 }
            else if c == 'P' { current_board[square][7] = 1 }
            else if c == 'N' { current_board[square][8] = 1 }
            else if c == 'B' { current_board[square][9] = 1 }
            else if c == 'R' { current_board[square][10] = 1 }
            else if c == 'Q' { current_board[square][11] = 1 }
            else if c == 'K' { current_board[square][12] = 1 }
            square += 1;
        } else {
            square += c.to_digit(10).unwrap() as usize;
        }
    }
    let mut current_board = current_board.into_iter().flatten().collect::<Vec<u8>>();
    current_board.push(current_color);
    return current_board;
}

fn board_array_from_json(j_input: Value, data_type: DataType) -> (Vec<u8>, Vec<f64>) {
    let mut eval_array: Vec<f64> = Vec::new();
    let mut board_array: Vec<u8> = Vec::new();
    let j: Vec<Value> = match j_input {
        Value::Array(j_array) => { j_array },
        _ => { panic!("invalid json input, must be array") },
    };
    for i in 0..j.len() {
        if i % (j.len() as f64 / 100.0) as usize == 0 {
            println!("{} % done", i as f64 / j.len() as f64 * 100 as f64);
        }

        let eval = match &j[i]["evals"][0]["pvs"][0] {
            Value::Object(eval_object) => { eval_object },
            _ => panic!("invalid json input, 2"),
        };
        match data_type { 
            DataType::Evals => {
                let current_eval = if eval.contains_key("cp") {
                    let eval_cp = match &eval["cp"] {
                        Value::Number(eval_cp) => { eval_cp },
                        _ => panic!("invalid json input, 3"),
                    };
                    adjusted_sigmoid(eval_cp.as_f64().expect("invalid json input, 3"))
                } else {
                    continue
                };
                eval_array.push(current_eval);
            } 
            DataType::Moves => {
                match &eval["line"] {
                    Value::String(line_string) => {
                        let mut line_chars = line_string.chars();
                        for _ in 0..2 {
                            let x: usize = line_chars.next().unwrap() as usize - 97;
                            let y: usize = line_chars.next().unwrap() as usize - 49;
                            let mut new_move = vec![0.0; 64];
                            new_move[y*8+x] = 1.0;
                            eval_array.extend(new_move);
                        }
                    },
                    _ => panic!("invalid json input, 3"),
                };
            }
        }

        let fen = match &j[i]["fen"] {
            Value::String(fen) => { fen },
            _ => { panic!("invalid json input, 4") },
        };
        let current_board = board_array_from_fen(fen.as_str());
        board_array.extend(current_board);
    }
    return (board_array, eval_array);
}

fn make_eval_dataset(data_type: DataType) -> std::io::Result<()>{
    //println!("{}", array![[1,2,3]; 5]);
    let mut file = fs::File::create("array.npy")?;
    for data_file_index in 0..41 {
        println!("Current file {}", data_file_index);
        let contents = fs::read_to_string(format!("../data/data{}.json", data_file_index))
            .expect("Should have been able to read the file");
        let v: Value = serde_json::from_str(&contents).expect("big fail 2");
        let (board_array, eval_array) = board_array_from_json(v, data_type);
        //println!("{:?}", board_array);
        //npyz::to_file_1d("array.npy", arr)?;
        let mut out_buf = vec![];
        let mut writer = {
            npyz::WriteOptions::<u8>::new()
                .default_dtype()
                .shape(&[(board_array.len()/833).try_into().unwrap(), 833])
                .writer(&mut out_buf)
                .begin_nd()?
        };
        writer.extend(board_array)?;
        writer.finish()?;
        let mut out_buf2 = vec![];
        let mut writer2 = match data_type {
            DataType::Evals => {
                npyz::WriteOptions::<f64>::new()
                    .default_dtype()
                    .shape(&[eval_array.len().try_into().unwrap()])
                    .writer(&mut out_buf2)
                    .begin_nd()?
            }
            DataType::Moves => {
                npyz::WriteOptions::<f64>::new()
                    .default_dtype()
                    .shape(&[(eval_array.len()/(64*2)).try_into().unwrap(), 2, 64])
                    .writer(&mut out_buf2)
                    .begin_nd()?
            }
        };
        writer2.extend(eval_array)?;
        writer2.finish()?;
        file.write_all(&out_buf)?;
        file.write_all(&out_buf2)?;
    }
    Ok(())
}

fn main() -> std::io::Result<()> {
    make_eval_dataset(DataType::Evals)?;
    Ok(())
}
