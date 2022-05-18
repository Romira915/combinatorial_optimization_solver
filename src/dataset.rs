use std::{
    convert::TryFrom,
    io::Error,
    path::{Path, PathBuf},
};

use getset::Getters;
use ndarray::Array1;
use tokio::{
    fs::File,
    io::{AsyncBufRead, AsyncBufReadExt, BufReader, Lines, ReadBuf},
};

#[derive(Debug, Clone, Getters)]
pub struct KnapsackData {
    #[get = "pub"]
    data_name: String,
    #[get = "pub"]
    n: usize,
    #[get = "pub"]
    capacity: usize,
    #[get = "pub"]
    optimum_solution: usize,
    #[get = "pub"]
    #[get = "pub"]
    costs: Array1<usize>,
    #[get = "pub"]
    weights: Array1<usize>,
    #[get = "pub"]
    answer_labels: Array1<u8>,
}

pub async fn load_knapsack<P>(path: P) -> Result<Vec<KnapsackData>, Error>
where
    P: AsRef<Path>,
{
    let path = path.as_ref();
    let file = File::open(path).await?;
    let mut reader = BufReader::new(file);
    let mut buf = String::new();

    let mut lines = reader.lines();

    let mut knapsack_datas = Vec::new();

    while let Ok(kd) = parse_knapsack(&mut lines).await {
        knapsack_datas.push(kd);
    }

    Ok(knapsack_datas)
}

async fn parse_knapsack<R>(lines: &mut Lines<R>) -> Result<KnapsackData, Error>
where
    R: AsyncBufRead + Unpin,
{
    let data_name = {
        let data_name;
        loop {
            let line = lines.next_line().await?;
            match line {
                Some(line) => {
                    if line.trim().is_empty() {
                        continue;
                    } else {
                        data_name = line;
                        break;
                    }
                }
                None => return Err(Error::new(std::io::ErrorKind::Other, "FIle EOF")),
            }
        }

        data_name
    };

    // metadata
    let n: usize = lines
        .next_line()
        .await?
        .unwrap()
        .split_whitespace()
        .last()
        .unwrap()
        .parse()
        .unwrap();
    let capacity: usize = lines
        .next_line()
        .await?
        .unwrap()
        .split_whitespace()
        .last()
        .unwrap()
        .parse()
        .unwrap();
    let optimum_solution = lines
        .next_line()
        .await?
        .unwrap()
        .split_whitespace()
        .last()
        .unwrap()
        .parse::<usize>();
    let _ = lines.next_line().await?.unwrap();

    let mut costs: Vec<usize> = Vec::new();
    let mut weights: Vec<usize> = Vec::new();
    let mut answer_labels: Vec<u8> = Vec::new();

    // input from column
    for _ in 0..n {
        let data = lines.next_line().await?.unwrap().replace(" ", "");
        let data = data.split(',').collect::<Vec<&str>>();

        costs.push(data[1].parse().unwrap());
        weights.push(data[2].parse().unwrap());
        answer_labels.push(data[3].parse().unwrap());
    }

    // End of point
    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() {
            continue;
        } else if line.trim().starts_with('-') {
            break;
        }
    }

    let costs = Array1::from_vec(costs);
    let weights = Array1::from_vec(weights);
    let answer_labels = Array1::from_vec(answer_labels);

    let optimum_solution = match optimum_solution {
        Ok(opt) => opt,
        Err(_) => answer_labels.map(|a| *a as usize).dot(&costs),
    };

    Ok(KnapsackData {
        data_name,
        n,
        capacity,
        optimum_solution,
        costs,
        weights,
        answer_labels,
    })
}
