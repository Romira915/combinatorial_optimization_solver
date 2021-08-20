use std::{
    convert::TryFrom,
    fs::File,
    io::{self, BufRead, BufReader},
    path::Path,
};

use ndarray::{Array2, Array4};
use num_traits::Pow;
use tokio::net::ToSocketAddrs;

use crate::model::QuboModel;

#[derive(Debug)]
pub struct TspNode {
    data_name: String,
    dim: usize,
    node: Vec<(f64, f64)>,
    opt: Option<Vec<usize>>,
    opt_len: Option<usize>,
}

impl TspNode {
    fn distance(&self, a: usize, b: usize) -> f64 {
        let x_dist = (self.node[a].0 - self.node[b].0).abs();
        let y_dist = (self.node[a].1 - self.node[b].1).abs();

        (x_dist + y_dist).sqrt()
    }

    fn try_read_opt_file(&mut self, path: &Path) -> Result<(), String> {
        let file = File::open(path).map_err(|e| e.to_string())?;
        let reader = BufReader::new(file);
        let lines = reader.lines();
        let mut opt = Vec::new();

        for line in lines {
            let line = line.map_err(|e| e.to_string())?;
            let split = line.split_whitespace().collect::<Vec<&str>>();

            match split[0] {
                "COMMENT" => {
                    self.opt_len = split.last().map(|s| s.parse().unwrap());
                }
                n => {
                    if let Ok(n) = n.parse() {
                        if n as isize != -1 {
                            opt.push(n);
                        }
                    }
                }
            }
        }

        if self.dim == opt.len() {
            self.opt = Some(opt);
        } else {
            return Err("dimension error".to_string());
        }

        Ok(())
    }
}

impl TryFrom<&str> for TspNode {
    type Error = String;
    fn try_from(url: &str) -> Result<Self, Self::Error> {
        let mut opt_path = std::path::PathBuf::from(&url);
        opt_path.set_extension("opt.tour");
        let err_message = "Failed to parse tsp file".to_string();
        let file = File::open(&url).map_err(|e| e.to_string())?;
        let reader = BufReader::new(file);
        let lines = reader.lines();

        let mut data_name = None;
        let mut dim = None;
        let mut node = Vec::new();

        for line in lines {
            let line = line.map_err(|e| e.to_string())?;
            let split = line.split_whitespace().collect::<Vec<&str>>();
            match split[0] {
                "NAME:" => {
                    data_name = split.last().map(|s| s.to_string());
                }
                "DIMENSION:" => {
                    dim = split
                        .last()
                        .map(|s| s.parse::<usize>().expect("Failed to usize parse"));
                }
                n => {
                    if let Ok(_) = n.parse::<usize>() {
                        let x = split[1].parse().unwrap();
                        let y = split[2].parse().unwrap();
                        node.push((x, y));
                    }
                }
            }
        }

        if let (Some(data_name), Some(dim)) = (data_name, dim) {
            if dim == node.len() {
                let mut node = TspNode {
                    data_name,
                    dim,
                    node,
                    opt: None,
                    opt_len: None,
                };
                node.try_read_opt_file(opt_path.as_path())?;
                Ok(node)
            } else {
                Err(err_message)
            }
        } else {
            Err(err_message)
        }
    }
}

impl From<TspNode> for QuboModel {
    fn from(tsp: TspNode) -> Self {
        let mut Q = Array4::<f64>::zeros((tsp.dim, tsp.dim, tsp.dim, tsp.dim));
        let mut indices = Vec::new();
        for i in 0..tsp.dim {
            for j in 0..tsp.dim {
                for u in 0..tsp.dim {
                    for v in 0..tsp.dim {
                        indices.push((i, j, u, v));
                    }
                }
            }
        }

        for (i, j, u, v) in indices {
            Q[[i, j, u, v]] += tsp.distance(u, v);
        }

        QuboModel::new(Array2::zeros((0, 0)))
    }
}
