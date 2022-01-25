use core::num;
use std::env;
use std::sync::Arc;

use combinatorial_optimization_solver::model::{IsingModel, QuboModel};
use combinatorial_optimization_solver::opt::TspNode;
use combinatorial_optimization_solver::solver::annealing_scheduler::AnnealingScheduler;
use combinatorial_optimization_solver::solver::simulated_annealing::SimulatedAnnealing;
use combinatorial_optimization_solver::solver::simulated_quantum_annealing::SimulatedQuantumAnnealing;
use combinatorial_optimization_solver::solver::{Solver, SolverVariant};
use combinatorial_optimization_solver::webhook::Webhook;
use ndarray::{array, Array1, Array2, ArrayView};
use ndarray_linalg::Scalar;
use rand::distributions::Uniform;
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};
use serenity::model::channel::Embed;
use std::convert::TryFrom;
use tokio::time::Instant;

fn number_partitioning(rng: &mut StdRng) -> Arc<IsingModel> {
    let n = 1000;
    let numbers = (rng)
        .sample_iter(Uniform::new(0., 1.))
        .take(n)
        .collect::<Vec<f64>>();
    let sum: f64 = numbers.iter().sum();
    let m = sum * 0.5;
    let J = {
        let mut J = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                if i < j {
                    J[[i, j]] = (numbers[i] * numbers[j]) as f32;
                }
            }
        }
        J /= 2.;
        J
    };
    let h = {
        let mut h = Array1::zeros(n);
        let k = sum / 2. - m;
        for i in 0..n {
            h[i] = (k * numbers[i]) as f32;
        }
        h
    };
    let ising = Arc::new(IsingModel::new(J, h));

    ising
}

fn tsp_ising(rng: &mut StdRng) -> (TspNode, Arc<IsingModel>, f32, f64) {
    let mut tsp = TspNode::try_from("./dataset/ulysses16.tsp").unwrap();
    let max_dist = tsp.max_distance() as f32;
    let bias = 0.15;
    // tsp.set_bias(max_dist * bias);
    tsp.set_bias(6.);
    let qubo = QuboModel::from(tsp.clone());
    let ising = IsingModel::from(qubo);
    let ising = Arc::new(ising);

    (tsp, ising, max_dist, bias)
}

#[tokio::main]
async fn main() {
    dotenv::dotenv().expect("Not found .env file");
    let url = env::var("WEBHOOK_URL").unwrap();

    let mut rng = rand::rngs::StdRng::from_rng(rand::thread_rng()).unwrap();

    let (tsp, ising, max_dist, bias) = tsp_ising(&mut rng);

    let steps = 1e5 as usize;
    let try_number_of_times = 10;
    let range_param_start = 3.;
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
            1.,
            steps,
            1,
            Arc::clone(&ising),
            None,
        )),
        SolverVariant::Sqa(SimulatedQuantumAnnealing::new(
            range_param_start,
            range_param_end,
            0.025,
            steps,
            40,
            Arc::clone(&ising),
            None,
        )),
        SolverVariant::Sqa(SimulatedQuantumAnnealing::new(
            range_param_start,
            range_param_end,
            0.0125,
            steps,
            80,
            Arc::clone(&ising),
            None,
        )),
        SolverVariant::Sqa(SimulatedQuantumAnnealing::new(
            range_param_start,
            range_param_end,
            0.00625,
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
            let best_len = tsp.len_from_state(ar.best_state.view());
            let best_len = match best_len {
                Ok(len) => len.to_string(),
                Err((len, message)) => format!("{} ({})", len, &message),
            };
            let len_vec = {
                let mut vec = Vec::new();
                for state in &ar.states {
                    let len = match tsp.len_from_state(state.bits.view()) {
                        Ok(len) => len.to_string(),
                        Err((len, message)) => format!("{} ({})", len, &message),
                    };
                    vec.push(len);
                }
                let mut str = "[".to_string();
                for len in vec {
                    str.push_str(&format!("{}, ", &len));
                }
                str.push_str("]");
                str
            };
            fields.push((
                format!("{}\nparameter {}", ar.solver_name, ar.parameter),
                format!(
                    "[best {}; ave {}; worst {}; best_len {}]\nbits {}\nlen {:?}",
                    ar.best_energy,
                    ar.average_energy,
                    ar.worst_energy,
                    best_len,
                    // ar.best_state,
                    "廃止",
                    len_vec
                ),
                false,
            ));
        }
        let strict_solution = tsp.opt_len().unwrap_or_default();
        e.title("Result")
            .description(format!(
                "dataset {}; try_number_of_times {}; 厳密解 {}; bias {} * {}; 実行時間 {:?}",
                tsp.data_name(),
                &try_number_of_times,
                strict_solution,
                max_dist,
                bias,
                end
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
