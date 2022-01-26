use combinatorial_optimization_solver::opt::TspNode;
use ndarray::Array1;

#[test]
fn tsp_node_test() {
    let node = vec![(3.0, 2.0), (4.0, 5.0), (10.0, 1.0), (8.0, 10.0)];
    let tsp_node = TspNode::new("test data".to_string(), node);
    let state_ok = Array1::from_vec(vec![1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
    let state_err = Array1::from_vec(vec![1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
    let len_ok = tsp_node.len_from_state(state_ok.view());
    let len_err = tsp_node.len_from_state(state_err.view());

    assert!(len_ok.is_ok());
    assert!(len_err.is_err());
}
