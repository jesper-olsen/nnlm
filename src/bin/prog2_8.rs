use gnuplot::{AxesCommon, Figure};
use nalgebra::{DMatrix, DVector};
use nnlm::*;

fn plot_mse(mse: &[f64], title: &str) {
    let epochs: Vec<i32> = (0..mse.len()).map(|i| i as i32).collect();
    let mut fg = Figure::new();
    fg.axes2d()
        .set_title(title, &[])
        .set_x_label("Epoch", &[])
        .set_y_label("MSE", &[])
        .lines(
            &epochs,
            mse,
            &[gnuplot::Caption("MSE"), gnuplot::Color("black")],
        );
    fg.show().unwrap();
}

fn calc_mean(data: &[[f64; 3]]) -> Vec<f64> {
    let mean: Vec<f64> = data.iter().fold(vec![0.0; 3], |acc, x| {
        acc.iter().zip(x.iter()).map(|(a, b)| a + b).collect()
    });
    mean.iter().map(|x| x / data.len() as f64).collect()
}

fn normalise_mean(data: &mut [[f64; 3]], mean: &[f64]) {
    for sample in data {
        for (value, &mean_val) in sample.iter_mut().zip(mean.iter()) {
            *value -= mean_val;
        }
    }
}

fn calc_max_vals(data: &[[f64; 3]]) -> Vec<f64> {
    data.iter().fold(vec![0.0; 3], |acc, x| {
        acc.iter()
            .zip(x.iter())
            .map(|(a, b)| a.max(b.abs()))
            .collect()
    })
}

fn normalise_max_val(data: &mut [[f64; 3]], max_vals: &[f64]) {
    const EPSILON: f64 = 1e-10;
    for sample in data {
        for (value, &max_val) in sample.iter_mut().zip(max_vals.iter()) {
            if max_val > EPSILON {
                *value /= max_val
            } else {
            }
        }
    }
}

fn main() {
    for dist in [-4.0, 1.0, 0.0, 4.0] {
        let central_radius = 10.0;
        let radius_variation = 6.0;
        let (data, labels) = halfmoons::<3000>(central_radius, radius_variation, dist);
        let mut data: Vec<_> = data
            .into_iter()
            .map(|[x, y]| [1.0, x, y]) // Add 1.0 - gives the perceptron a bias term
            .collect();

        ///////
        const N: usize = 2;

        let mean = calc_mean(&data[..1000]);
        normalise_mean(&mut data[..1000], &mean);
        normalise_mean(&mut data[1000..], &mean);

        let max_vals = calc_max_vals(&data[..1000]);
        normalise_max_val(&mut data[..1000], &max_vals);
        normalise_max_val(&mut data[1000..], &max_vals);

        // Step 2: Calculate weights using least squares (LS)
        let mut a = DMatrix::zeros(1000, 2);
        let mut yy = DVector::zeros(1000);

        for i in 0..1000 {
            a[(i, 0)] = data[i][1];
            a[(i, 1)] = data[i][2];
            yy[i] = labels[i] as f64;
        }

        // LS solution: w = inv(A'A + mu*I) * A'yy
        let mu = 0.0;
        let mu_identity = DMatrix::identity(2, 2) * mu; // mu * I
        let a_t = a.transpose(); // Transpose of A
        let w = (a_t.clone() * a + mu_identity).try_inverse().unwrap() * a_t * yy; // Use try_inverse()

        let mut weights = [0.0f64; 3];
        weights[1] = w[0];
        weights[2] = w[1];

        let correct: usize = data[1000..]
            .iter()
            .zip(labels[1000..].iter())
            .map(|(x, d)| { x.iter().zip(weights.iter()).map(|(xi,wi)| xi*wi).sum::<f64>().signum() as i8 == *d})
            .filter(|f| *f)
            .count();


        let error = 1.0-correct as f64/data[1000..].len() as f64;
        let error = 100.0*error;
        let title = format!(
            "Least Squares Classification with Half-Moon Data - dist: {dist}; Error: {error:.1}%"
        );
        plot(&data[1000..], &weights, &title);
    }
}
