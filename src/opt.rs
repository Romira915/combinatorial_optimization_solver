use std::{
    convert::TryFrom,
    fs::File,
    io::{self, BufRead, BufReader},
    path::Path,
};

use getset::{Getters, Setters};
use ndarray::{Array1, Array2, Array4, ArrayView1};
use num_traits::{Pow, ToPrimitive, Zero};
use tokio::net::ToSocketAddrs;

use crate::model::QuboModel;

#[derive(Debug, Clone, Getters, Setters)]
pub struct TspNode {
    #[get = "pub"]
    data_name: String,
    #[get = "pub"]
    dim: usize,
    #[get = "pub"]
    node: Vec<(f64, f64)>,
    #[get = "pub"]
    opt: Option<Vec<usize>>,
    #[get = "pub"]
    #[set = "pub"]
    bias: f32,
}

impl TspNode {
    pub fn distance(&self, a: usize, b: usize) -> f64 {
        let x_dist = (self.node[a].0 - self.node[b].0).abs();
        let y_dist = (self.node[a].1 - self.node[b].1).abs();

        (x_dist + y_dist).sqrt()
    }

    pub fn opt_len(&self) -> Option<f64> {
        if let Some(opt) = &self.opt {
            let mut len = 0.;
            for (i, node) in opt.iter().enumerate() {
                len += self.distance(*node, opt[(i + 1) % opt.len()]);
            }

            Some(len)
        } else {
            None
        }
    }

    fn try_read_opt_file(&mut self, path: &Path) -> Result<(), String> {
        let file = File::open(path).map_err(|e| e.to_string())?;
        let reader = BufReader::new(file);
        let lines = reader.lines();
        let mut opt = Vec::new();

        for line in lines {
            let line = line.map_err(|e| e.to_string())?;
            let split = line.trim().split_whitespace().collect::<Vec<&str>>();
            if split.len().is_zero() {
                continue;
            }

            match split[0] {
                "COMMENT" => {}
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

    pub fn len_from_state(&self, state: ArrayView1<i8>) -> Result<f64, (f64, String)> {
        let mut traveling_order = Vec::new();
        for (i, n) in state.iter().enumerate() {
            if *n == 1 {
                traveling_order.push((i % self.dim, i / self.dim));
            }
        }
        traveling_order.sort();

        let (len, _, _, satisfies_constraint) = traveling_order.iter().fold(
            (0., 0usize, 0usize, true),
            |(len, pre_order, pre_city_number, sc), (order, city_number)| {
                (
                    len + self.distance(pre_city_number, *city_number),
                    *order,
                    *city_number,
                    pre_order + 1 == *order && sc,
                )
            },
        );

        if satisfies_constraint {
            Ok(len)
        } else {
            Err((len, "制約条件エラー".to_string()))
        }
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
            let split = line.trim().split_whitespace().collect::<Vec<&str>>();
            if split.len().is_zero() {
                continue;
            }
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
                    bias: 1.,
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
        // let mut Q = Array4::<f32>::zeros((tsp.dim, tsp.dim, tsp.dim, tsp.dim));
        let mut Q = Array2::zeros((tsp.dim.pow(2), tsp.dim.pow(2)));
        for u in 0..tsp.dim {
            for v in 0..tsp.dim {
                for i in 0..tsp.dim {
                    for j in 0..tsp.dim {
                        let ui = u * tsp.dim + i;
                        let vj = v * tsp.dim + j;
                        let k = (ui as isize - vj as isize).abs() as usize;

                        if ui > vj {
                            continue;
                        }
                        if ui == vj {
                            Q[[ui, vj]] -= tsp.bias() * 2.;
                        }
                        if u == v && i != j {
                            Q[[ui, vj]] += tsp.bias() * 2.;
                        }
                        if u < v && i == j {
                            Q[[ui, vj]] += tsp.bias() * 2.;
                        }

                        if (k == 1 || k == tsp.dim - 1) && u < v {
                            for r in 0..(tsp.dim.pow(2)) {
                                Q[[ui, vj]] += tsp.distance(u, v) as f32;
                            }
                        }
                    }
                }
            }
        }

        // let Q = Q.into_shape((tsp.dim.pow(2), tsp.dim.pow(2))).unwrap();
        QuboModel::new(Q)
    }
}
