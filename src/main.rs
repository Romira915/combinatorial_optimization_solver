use core::num;
use std::env;
use std::sync::Arc;

use combinatorial_optimization_solver::model::{clone_array_row_matrix, IsingModel, QuboModel};
use combinatorial_optimization_solver::opt::TspNode;
use combinatorial_optimization_solver::solver::annealing_scheduler::AnnealingScheduler;
use combinatorial_optimization_solver::solver::simulated_annealing::SimulatedAnnealing;
use combinatorial_optimization_solver::solver::simulated_quantum_annealing::SimulatedQuantumAnnealing;
use combinatorial_optimization_solver::solver::{Solver, SolverVariant};
use combinatorial_optimization_solver::webhook::Webhook;
use ndarray::{array, Array1, Array2, ArrayBase, ArrayView, ArrayView1, Dim, OwnedRepr};
use ndarray_linalg::Scalar;
use rand::distributions::Uniform;
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};
use serenity::model::channel::Embed;
use std::convert::TryFrom;
use tokio::time::Instant;

fn number_partitioning(
    n: usize,
    rate: f64,
    rng: &mut StdRng,
) -> (Array1<f64>, f64, Arc<IsingModel>, f64) {
    let numbers = (rng)
        .sample_iter(Uniform::new(0., 1000.))
        .take(n)
        .collect::<Vec<f64>>();
    let sum: f64 = numbers.iter().sum();
    let m = sum * rate;
    let J = {
        let mut J = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                J[[i, j]] = numbers[i] * numbers[j];
            }
        }
        J /= 4.;
        J
    };
    let h = {
        let mut h = Array1::zeros(n);
        let k = sum / 2. - m;
        for i in 0..n {
            h[i] = k * numbers[i];
        }
        h
    };
    let ising = Arc::new(IsingModel::new(J, h));

    let numbers = Array1::from(numbers);
    let constant = (numbers.sum() / 2. - m).powf(2.);

    (numbers, m, ising, constant)
}

fn tsp_ising(rng: &mut StdRng) -> (TspNode, Arc<IsingModel>, f64, f64) {
    let mut tsp = TspNode::try_from("./dataset/ulysses16.tsp").unwrap();
    let max_dist = tsp.max_distance();
    let bias = 0.6;
    tsp.set_bias(max_dist * bias);
    // tsp.set_bias(15.);
    let qubo = QuboModel::from(tsp.clone());
    let ising = IsingModel::from(qubo);
    let ising = Arc::new(ising);

    (tsp, ising, max_dist, bias)
}

fn knapsack(
    n: usize,
    capacity: u32,
    rng: &mut StdRng,
) -> (Arc<IsingModel>, Array1<f64>, Array1<f64>) {
    let mut cost = rng
        .sample_iter(Uniform::new(1, 10000))
        .take(n)
        .collect::<Array1<u32>>();
    let mut weight = rng
        .sample_iter(Uniform::new(50, (capacity as f64 * 1.3) as u32))
        .take(n)
        .collect::<Array1<u32>>();

    let max_c = cost.iter().max().unwrap().to_owned();

    let Q = {
        let mut Q = Array2::zeros((n, n));
        let A = max_c as f64 * 1.1;

        for i in 0..n {
            for j in 0..n {
                Q[[i, j]] += weight[i] as f64 * weight[i] as f64;

                if i == j {
                    Q[[i, j]] += -2. * A * capacity as f64 * weight[i] as f64;
                    Q[[i, j]] += -2. * cost[i] as f64;
                }
            }
        }

        Q
    };

    let qubo = QuboModel::new(Q);
    let ising = IsingModel::from(qubo);
    let ising = Arc::new(ising);
    let cost = cost.map(|i| *i as f64);
    let weight = weight.map(|i| *i as f64);

    (ising, cost, weight)
}

#[tokio::main]
async fn main() {
    dotenv::dotenv().expect("Not found .env file");
    let url = env::var("WEBHOOK_URL").unwrap();

    let mut rng = rand::rngs::StdRng::from_rng(rand::thread_rng()).unwrap();

    // let (tsp, ising, max_dist, bias) = tsp_ising(&mut rng);
    let n = 20;
    let capacity = 500;
    let (ising, cost, weight) = knapsack(n, capacity, &mut rng);

    let steps = 3e5 as usize;
    let try_number_of_times = 30;
    let range_param_start = 1.;
    let range_param_end = 1e-06;
    let solvers = vec![
        SolverVariant::Sa(SimulatedAnnealing::new(
            range_param_start,
            range_param_end,
            steps,
            Arc::clone(&ising),
            None,
        )),
        SolverVariant::Sqa(SimulatedQuantumAnnealing::new(
            range_param_start,
            range_param_end,
            1. / 1.,
            steps,
            1,
            Arc::clone(&ising),
            None,
        )),
        SolverVariant::Sqa(SimulatedQuantumAnnealing::new(
            range_param_start,
            range_param_end,
            1. / 40.,
            steps,
            40,
            Arc::clone(&ising),
            None,
        )),
        SolverVariant::Sqa(SimulatedQuantumAnnealing::new(
            range_param_start,
            range_param_end,
            1. / 80.,
            steps,
            80,
            Arc::clone(&ising),
            None,
        )),
        SolverVariant::Sqa(SimulatedQuantumAnnealing::new(
            range_param_start,
            range_param_end,
            1. / 160.,
            steps,
            160,
            Arc::clone(&ising),
            None,
        )),
        SolverVariant::Sqa(SimulatedQuantumAnnealing::new(
            range_param_start,
            range_param_end,
            1. / 320.,
            steps,
            320,
            Arc::clone(&ising),
            None,
        )),
        SolverVariant::Sqa(SimulatedQuantumAnnealing::new(
            range_param_start,
            range_param_end,
            1. / 640.,
            steps,
            640,
            Arc::clone(&ising),
            None,
        )),
        SolverVariant::Sqa(SimulatedQuantumAnnealing::new(
            range_param_start,
            range_param_end,
            1. / 1280.,
            steps,
            1280,
            Arc::clone(&ising),
            None,
        )),
    ];

    let scheduler = AnnealingScheduler::new(solvers, try_number_of_times);

    let start = Instant::now();
    let records = scheduler.run();
    let analysis_records = AnnealingScheduler::analysis(&records);

    let end = start.elapsed();

    let embed = Embed::fake(move |e| {
        let mut fields = Vec::new();
        for ar in &analysis_records {
            // let best_len = tsp.len_from_state(ar.best_state.view());
            // let best_len = match best_len {
            //     Ok(len) => len.to_string(),
            //     Err((len, message)) => format!("{} ({})", len, &message),
            // };
            let best_state = Array1::from_iter(ar.best_state.iter().map(|s| s.to_owned() as f64));
            let best_cost = best_state.dot(&cost);
            let best_weight = best_state.dot(&weight);

            let worst_state = Array1::from_iter(ar.worst_state.iter().map(|s| s.to_owned() as f64));
            let worst_cost = worst_state.dot(&cost);
            let worst_weight = worst_state.dot(&weight);

            fields.push((
                format!("{}\nparameter {}", ar.solver_name, ar.parameter),
                format!(
                    "[best E {} cost {} weight {}; ave {}; worst E {} cost {} weight {}; \nbits {}\n",
                    ar.best_energy,
                    best_cost,
                    best_weight,
                    ar.average_energy,
                    ar.worst_energy,
                    worst_cost,
                    worst_weight,
                    // best_state,
                    "廃止",
                ),
                false,
            ));
            println!("{:?}", &fields);
        }
        e.title("Result")
            .description(format!(
                "try_number_of_times {}; n {}; capacity {}; 実行時間 {:?}",
                &try_number_of_times, &n, &capacity, end
            ))
            .fields(fields)
    });

    let webhook = Webhook::new(&url);
    webhook.send(embed).await.unwrap();

    //     for record in &records {
    //         let cost = record
    //             .bits
    //             .map(|f| *f as f64)
    //             .dot(&Array1::from_iter(numbers.clone().into_iter()));

    //         println!("\n---- result ----");
    //         println!("param [m {}; sum {}]", &m, &sum);
    //         println!("{};\ncost {}", &record, (cost - m).abs());
    //     }
}
