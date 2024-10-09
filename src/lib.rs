use rand::prelude::SliceRandom;
use rand::Rng;

pub fn halfmoons(
    central_radius: f64,
    radius_variation: f64,
    dist: f64,
    n_samp: usize,
) -> (Vec<(f64,f64)>,Vec<(f64,f64)>) {
    // half doughnut
    // radius_variation: radius_variation of doughnut
    // central_radius: center radius of dounut - sampels are at rad += radius_variation/2

    if central_radius < radius_variation / 2.0 {
        panic!("The radius should be at least larger than half the radius_variation");
    }

    let mut rng = rand::thread_rng();
    let data1: Vec<(f64, f64)> = (0..(n_samp ))
        .map(|_| {
            (
                central_radius - radius_variation / 2.0 + radius_variation * rng.gen::<f64>(),
                std::f64::consts::PI * rng.gen::<f64>(),
            )
        })
        .map(|(radius, theta)| (radius * theta.cos(), radius * theta.sin()))
        .collect();

    let data2: Vec<(f64, f64)> = (0..(n_samp ))
        .map(|_| {
            (
                central_radius - radius_variation / 2.0 + radius_variation * rng.gen::<f64>(),
                std::f64::consts::PI * rng.gen::<f64>(),
            )
        })
        .map(|(radius, theta)| {
            (
                -radius * theta.cos() + central_radius,
                -radius * theta.sin() - dist,
            )
        })
        .collect();
    (data1,data2)
}


#[derive(Debug)]
pub struct HalfMoonData {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub label: Vec<i32>,
}

pub fn halfmoon(
    central_radius: f64,
    radius_variation: f64,
    dist: f64,
    n_samp: usize,
) -> HalfMoonData {
    // half doughnut
    // radius_variation: radius_variation of doughnut
    // central_radius: center radius of dounut - sampels are at rad += radius_variation/2

    if central_radius < radius_variation / 2.0 {
        panic!("The radius should be at least larger than half the radius_variation");
    }

    if n_samp % 2 != 0 {
        panic!("Please make sure the number of samples is even");
    }

    let mut x = Vec::with_capacity(n_samp);
    let mut y = Vec::with_capacity(n_samp);
    let mut label = Vec::with_capacity(n_samp);
    let mut rng = rand::thread_rng();

    // First half-moon
    for _ in 0..(n_samp / 2) {
        let radius =
            (central_radius - radius_variation / 2.0) + radius_variation * rng.gen::<f64>();
        let theta = std::f64::consts::PI * rng.gen::<f64>();
        x.push(radius * theta.cos());
        y.push(radius * theta.sin());
        label.push(1);
    }

    // Second half-moon (mirrored)
    for _ in 0..(n_samp / 2) {
        let radius =
            (central_radius - radius_variation / 2.0) + radius_variation * rng.gen::<f64>();
        let theta = std::f64::consts::PI * rng.gen::<f64>();
        x.push(radius * (-theta.cos()) + central_radius);
        y.push(radius * (-theta.sin()) - dist);
        label.push(-1);
    }

    // Shuffle data
    let mut indices: Vec<usize> = (0..n_samp).collect();
    indices.shuffle(&mut rng);
    let shuffled_x: Vec<f64> = indices.iter().map(|&i| x[i]).collect();
    let shuffled_y: Vec<f64> = indices.iter().map(|&i| y[i]).collect();
    let shuffled_label: Vec<i32> = indices.iter().map(|&i| label[i]).collect();

    HalfMoonData {
        x: shuffled_x,
        y: shuffled_y,
        label: shuffled_label,
    }
}

pub fn sign(value: f64) -> i32 {
    if value >= 0.0 {
        1
    } else {
        -1
    }
}

pub fn perceptron_train(
    data: &HalfMoonData,
    num_tr: usize,
    num_epochs: usize,
) -> Vec<f64> {
    let mut w = vec![0.0; 3]; // Initial weights, including bias

    for _epoch in 0..num_epochs {
        for i in 0..num_tr {
            let x = vec![1.0, data.x[i], data.y[i]]; // Input with bias
            let d = data.label[i]; // Desired output
            let y = sign(w.iter().zip(&x).map(|(wi, xi)| wi * xi).sum::<f64>());
            let error = (d - y) as f64;
            for j in 0..w.len() {
                let lr = 1.0; // Optional - impl annealing, e.g. linear decay
                w[j] +=  lr*error * x[j];
            }
        }
    }
    w
}

pub fn perceptron_test(data: &HalfMoonData, w: &[f64], num_te: usize, num_tr: usize) -> f64 {
    let mut errors = 0;
    for i in 0..num_te {
        let idx = num_tr + i;
        let x = vec![1.0, data.x[idx], data.y[idx]]; // Input with bias
        let y = sign(w.iter().zip(&x).map(|(wi, xi)| wi * xi).sum::<f64>());
        if y != data.label[idx] {
            errors += 1;
        }
    }
    errors as f64 / num_te as f64 * 100.0 // Return error rate in percentage
}

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
