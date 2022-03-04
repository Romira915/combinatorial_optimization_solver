use std::{
    convert::TryFrom,
    fs::File,
    io::{self, BufRead, BufReader},
    path::Path,
};

use getset::{Getters, Setters};
use ndarray::{Array1, Array2, Array4, ArrayView1};
use num_traits::{Float, Pow, ToPrimitive, Zero};
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
    bias: f64,
    max_dist: Option<f64>,
}

impl TspNode {
    pub fn new(data_name: String, node: Vec<(f64, f64)>) -> Self {
        TspNode {
            data_name,
            dim: node.len(),
            node,
            opt: None,
            bias: 1.0,
            max_dist: None,
        }
    }

    pub fn distance(&self, a: usize, b: usize) -> f64 {
        let x_dist = (self.node[a].0 - self.node[b].0).powf(2.);
        let y_dist = (self.node[a].1 - self.node[b].1).powf(2.);

        (x_dist + y_dist).sqrt()
    }

    pub fn max_distance(&mut self) -> f64 {
        match self.max_dist {
            Some(max) => max,
            None => {
                let mut max = 0.;
                for a in 0..self.dim {
                    for b in 0..self.dim {
                        max = max.max(self.distance(a, b));
                    }
                }
                self.max_dist = Some(max);

                max
            }
        }
    }

    pub fn opt_len(&self) -> Option<f64> {
        if let Some(opt) = &self.opt {
            let (len, _) = opt
                .iter()
                .fold((0., opt[0]), |(len, pre_city_number), city_number| {
                    (
                        len + self.distance(pre_city_number - 1, city_number - 1),
                        *city_number,
                    )
                });

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

    pub fn len_from_state(&self, state: ArrayView1<i8>) -> Result<f64, String> {
        let dim = (state.dim() as f64).sqrt() as usize;
        if dim != state.sum() as usize {
            return Err("制約条件エラー".to_string());
        }
        let matrix = state.to_shape((dim, dim)).unwrap();
        for row in matrix.rows() {
            if row.sum() != 1 {
                return Err("制約条件エラー".to_string());
            }
        }
        for column in matrix.columns() {
            if column.sum() != 1 {
                return Err("制約条件エラー".to_string());
            }
        }

        let mut traveling_order = Vec::new();
        for (i, n) in state.iter().enumerate() {
            if *n == 1 {
                // (order, city_num)
                traveling_order.push((i % self.dim, i / self.dim));
            }
        }
        traveling_order.sort();

        let (len, _, _, satisfies_constraint) = traveling_order.iter().fold(
            (0., 0usize, traveling_order[0].1, true),
            |(len, pre_order, pre_city_number, sc), (order, city_number)| {
                (
                    len + self.distance(pre_city_number, *city_number),
                    *order,
                    *city_number,
                    (pre_order + 1 == *order && sc) || *order == 0,
                )
            },
        );

        if satisfies_constraint {
            Ok(len)
        } else {
            Err("制約条件エラー".to_string())
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
                    max_dist: None,
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

        for i in 0..tsp.dim {
            for j in 0..tsp.dim {
                for a in 0..tsp.dim {
                    for b in 0..tsp.dim {
                        let ia = i + tsp.dim * a;
                        let jb = j + tsp.dim * b;
                        if i == j {
                            Q[[i + tsp.dim * a, ((i + 1) % tsp.dim) + tsp.dim * b]] +=
                                tsp.distance(a, b);
                            Q[[ia, jb]] += tsp.bias;
                        }

                        if i == j && a == b {
                            Q[[ia, jb]] += -2. * 2. * tsp.bias;
                        }

                        if a == b {
                            Q[[ia, jb]] += tsp.bias;
                        }
                    }
                }
            }
        }

        // let Q = Q.into_shape((tsp.dim.pow(2), tsp.dim.pow(2))).unwrap();
        QuboModel::new(Q)
    }
}
