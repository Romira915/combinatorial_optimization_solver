use std::array;

use combinatorial_optimization_solver::math::DiagonalMatrix;
use combinatorial_optimization_solver::model::{IsingModel, QuboModel};
use combinatorial_optimization_solver::opt::TspNode;
use ndarray::prelude::*;
use ndarray::{Array1, Array2};
use ndarray_linalg::{IntoTriangular, Scalar, UPLO};

#[test]
fn tsp_node_test() {
    let node = vec![(3.0, 2.0), (4.0, 5.0), (10.0, 1.0), (8.0, 10.0)];
    let tsp_node = TspNode::new("test data".to_string(), node);
    let state = Array1::from_vec(vec![1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
    let len = tsp_node.len_from_state(state.view());

    assert!(len.is_ok())
}

#[test]
fn ising_to_qubo_test() {
    let n = 4;
    // let q = array![[2, 5, 6, 7], [0, 4, 6, 4], [0, 0, 6, 4], [0, 0, 0, 4],];
    let q = array![[2, 5, 6, 7], [5, 4, 6, 4], [6, 6, 6, 4], [7, 4, 4, 4],];
    let q = q.map(|i| *i as f64);
    let qubo = QuboModel::new(q.clone());
    let ising = IsingModel::from(qubo.clone());

    let bin = array![1, 1, 0, 1];
    let spins = bin.map(IsingModel::SPINS_FROM_BITS);

    let e_qubo = qubo.calculate_energy(bin.view());
    let e_ising = ising.calculate_energy(spins.view());

    let offset = {
        let mut offset = 0.;
        for i in 0..n {
            for j in 0..n {
                offset += q[[i, j]];
                if i == j {
                    offset += q[[i, i]];
                }
            }
        }
        offset / 4.
    };

    let delta = (e_qubo - (e_ising + offset)).abs();
    println!("qubo {}", qubo.Q());
    println!("ising {}\n{}", ising.J(), ising.h());
    println!("delta {}", delta);

    assert!(delta < 1e-10);
}
