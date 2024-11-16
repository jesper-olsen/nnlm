use nalgebra::{DMatrix, DVector};
use nnlm::*;

// least squares with L2 regularisation  (mu)
fn least_squares(dist: f64, mu: f64) {
    let central_radius = 10.0;
    let radius_variation = 6.0;
    let (mut data, labels) = halfmoons::<3000>(central_radius, radius_variation, dist);
    const NTRAIN: usize = 1000;

    let mean = nnlm::calc_mean(&data[..NTRAIN]);
    normalise_mean(&mut data[..NTRAIN], &mean);
    normalise_mean(&mut data[NTRAIN..], &mean);

    let max_vals = calc_max(&data[..NTRAIN]);
    normalise_max(&mut data[..NTRAIN], &max_vals);
    normalise_max(&mut data[NTRAIN..], &max_vals);

    let data: Vec<_> = data
        .into_iter()
        .map(|[x, y]| [1.0, x, y]) // Add 1.0 - gives the perceptron a bias term
        .collect();

    // Calculate weights using least squares (LS)
    let mut a = DMatrix::zeros(NTRAIN, 2);
    let mut yy = DVector::zeros(NTRAIN);

    for i in 0..NTRAIN {
        a[(i, 0)] = data[i][1];
        a[(i, 1)] = data[i][2];
        yy[i] = labels[i] as f64;
    }

    // LS solution: w = inv(A'A + mu*I) * A'yy
    let mu_identity = DMatrix::identity(2, 2) * mu; // mu * I
    let a_t = a.transpose(); // Transpose of A
    let w = (a_t.clone() * a + mu_identity).try_inverse().unwrap() * a_t * yy; // Use try_inverse()

    let mut weights = [0.0f64; 3];
    weights[1] = w[0];
    weights[2] = w[1];

    let correct: usize = data[NTRAIN..]
        .iter()
        .zip(labels[NTRAIN..].iter())
        .map(|(x, d)| {
            x.iter()
                .zip(weights.iter())
                .map(|(xi, wi)| xi * wi)
                .sum::<f64>()
                .signum() as i8
                == *d
        })
        .filter(|f| *f)
        .count();

    let error = 1.0 - correct as f64 / data[NTRAIN..].len() as f64;
    let error = 100.0 * error;
    let title = format!(
        "Least Squares Classification with Half-Moon Data - dist: {dist}; L2-reg: {mu}; Error: {error:.1}%"
    );
    plot(&data[NTRAIN..], &weights, &title);
}

fn main() {
    for dist in [0.0] {
        for mu in [0.0, 0.1] {
            least_squares(dist, mu);
        }
    }
}
