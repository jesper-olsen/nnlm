use gnuplot::{AxesCommon, Color, Figure, PointSymbol};
//use rand::Rng;
use stmc_rs::marsaglia::Marsaglia;
pub mod gmm;
pub mod kernel;
pub mod perceptron;
pub mod rbf;

/// Halfmoons: randomly samples two "half moons". Returns samples and their labels. Samples are 3D:
/// 1 (bias) + 2d coordinates
pub fn halfmoons<const NSAMP: usize>(
    central_radius: f64,
    radius_variation: f64,
    dist: f64,
) -> ([[f64; 2]; NSAMP], [i8; NSAMP]) {
    // half doughnut
    // radius_variation: diameter of doughnut ring
    // central_radius: center radius of dounut - sampels are at rad += radius_variation/2

    if central_radius < radius_variation / 2.0 {
        panic!("The central_radius should be at least larger than half the radius_variation");
    }

    let mut rng = Marsaglia::new(12, 34, 56, 78);
    let mut data = [[0.0; 2]; NSAMP];
    let mut labels = [0; NSAMP];
    for i in (0..NSAMP).step_by(2) {
        let radius = central_radius - radius_variation / 2.0 + radius_variation * rng.uni();
        let theta = std::f64::consts::PI * rng.uni();
        data[i] = [radius * theta.cos(), radius * theta.sin()];
        labels[i] = 1;

        let radius = central_radius - radius_variation / 2.0 + radius_variation * rng.uni();
        let theta = std::f64::consts::PI * rng.uni();
        data[i + 1] = [
            -radius * theta.cos() + central_radius,
            -radius * theta.sin() - dist,
        ];
        labels[i + 1] = -1;
    }

    (data, labels)
}

pub fn calc_mean(data: &[[f64; 2]]) -> Vec<f64> {
    let mean: Vec<f64> = data.iter().fold(vec![0.0; 3], |acc, x| {
        acc.iter().zip(x.iter()).map(|(a, b)| a + b).collect()
    });
    mean.iter().map(|x| x / data.len() as f64).collect()
}

pub fn normalise_mean(data: &mut [[f64; 2]], mean: &[f64]) {
    for sample in data {
        for (value, &mean_val) in sample.iter_mut().zip(mean.iter()) {
            *value -= mean_val;
        }
    }
}

pub fn calc_max(data: &[[f64; 2]]) -> Vec<f64> {
    data.iter().fold(vec![f64::MIN; 2], |acc, x| {
        acc.iter()
            .zip(x.iter())
            .map(|(a, b)| a.max(b.abs()))
            .collect()
    })
}

pub fn normalise_max(data: &mut [[f64; 2]], max_vals: &[f64]) {
    const EPSILON: f64 = 1e-10;
    for sample in data {
        for (value, &max_val) in sample.iter_mut().zip(max_vals.iter()) {
            if max_val > EPSILON {
                *value /= max_val
            }
        }
    }
}

pub fn plot_mse(mse: &[f64], title: &str) {
    let epochs: Vec<i32> = (0..mse.len()).map(|i| i as i32).collect();
    let mut fg = Figure::new();
    fg.axes2d()
        .set_title(title, &[])
        .set_x_label("Epoch", &[])
        .set_y_label("MSE", &[])
        .lines(
            &epochs,
            mse,
            &[gnuplot::Caption("MSE"), gnuplot::Color(gnuplot::RGBString("black"))],
        );
    fg.show().unwrap();
}

pub fn plot(data: &[[f64; 3]], weights: &[f64], title: &str) {
    let (xmin, xmax) = data
        .iter()
        .map(|&v| v[1])
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), val| {
            (f64::min(min, val), f64::max(max, val))
        });
    let (ymin, ymax) = data
        .iter()
        .map(|&v| v[2])
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), val| {
            (f64::min(min, val), f64::max(max, val))
        });

    // Grid points - different colours and different sides of decision boundary
    let mut grid = Vec::new();

    let x_range = xmax - xmin;
    let y_range = ymax - ymin;
    let step_size = f64::min(x_range, y_range) / 150.0; // grid density

    let mut x = xmin;
    while x <= xmax {
        let mut y = ymin;
        while y <= ymax {
            let input = [1.0, x, y]; // input with bias
            let z: f64 = weights.iter().zip(input.iter()).map(|(w, x)| w * x).sum();
            grid.push((x, y, z.signum() as i8));
            y += step_size
        }
        x += step_size
    }

    let pos_x: Vec<f64> = grid
        .iter()
        .filter(|(_, _, z)| *z == 1)
        .map(|(x, _, _)| *x)
        .collect();
    let pos_y: Vec<f64> = grid
        .iter()
        .filter(|(_, _, z)| *z == 1)
        .map(|(_, y, _)| *y)
        .collect();
    let neg_x: Vec<f64> = grid
        .iter()
        .filter(|(_, _, z)| *z == -1)
        .map(|(x, _, _)| *x)
        .collect();
    let neg_y: Vec<f64> = grid
        .iter()
        .filter(|(_, _, z)| *z == -1)
        .map(|(_, y, _)| *y)
        .collect();

    // calculate decision boundary
    let x_vals: Vec<f64> = (0..=100)
        .map(|i| xmin + (xmax - xmin) * (i as f64 / 100.0))
        .collect();
    let y_vals: Vec<f64> = x_vals
        .iter()
        .map(|&x| -(weights[0] + weights[1] * x) / weights[2])
        .collect();

    let vx: Vec<f64> = data.iter().map(|v| v[1]).collect();
    let vy: Vec<f64> = data.iter().map(|v| v[2]).collect();
    // Plot using gnuplot
    let mut fg = Figure::new();
    fg.axes2d()
        .set_title(title, &[])
        .points(&vx, &vy, &[PointSymbol('O'), Color(gnuplot::RGBString("blue"))]) // Original data points
        .points(&pos_x, &pos_y, &[PointSymbol('.'), Color(gnuplot::RGBString("green"))]) // Positive classified points
        .points(&neg_x, &neg_y, &[PointSymbol('.'), Color(gnuplot::RGBString("red"))]) // Negative classified points
        .lines(&x_vals, &y_vals, &[Color(gnuplot::RGBString("black"))]);
    fg.show().unwrap();
}

pub fn plot_mesh<F>(data: &[[f64; 3]], model: F, title: &str)
where
    F: Fn(&[f64; 3]) -> f64,
{
    let (xmin, xmax) = data
        .iter()
        .map(|&v| v[1])
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), val| {
            (f64::min(min, val), f64::max(max, val))
        });
    let (ymin, ymax) = data
        .iter()
        .map(|&v| v[2])
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), val| {
            (f64::min(min, val), f64::max(max, val))
        });

    // Grid points - different colours and different sides of decision boundary
    let mut grid = Vec::new();

    let x_range = xmax - xmin;
    let y_range = ymax - ymin;
    let step_size = f64::min(x_range, y_range) / 150.0; // grid density

    let mut x = xmin;
    while x <= xmax {
        let mut y = ymin;
        while y <= ymax {
            let input = [1.0, x, y]; // input with bias
                                     //let z: f64 = weights.iter().zip(input.iter()).map(|(w, x)| w * x).sum();
            let z = model(&input);
            grid.push((x, y, z.signum() as i8));
            y += step_size
        }
        x += step_size
    }

    let pos_x: Vec<f64> = grid
        .iter()
        .filter(|(_, _, z)| *z == 1)
        .map(|(x, _, _)| *x)
        .collect();
    let pos_y: Vec<f64> = grid
        .iter()
        .filter(|(_, _, z)| *z == 1)
        .map(|(_, y, _)| *y)
        .collect();
    let neg_x: Vec<f64> = grid
        .iter()
        .filter(|(_, _, z)| *z == -1)
        .map(|(x, _, _)| *x)
        .collect();
    let neg_y: Vec<f64> = grid
        .iter()
        .filter(|(_, _, z)| *z == -1)
        .map(|(_, y, _)| *y)
        .collect();

    let vx: Vec<f64> = data.iter().map(|v| v[1]).collect();
    let vy: Vec<f64> = data.iter().map(|v| v[2]).collect();
    // Plot using gnuplot
    let mut fg = Figure::new();
    fg.axes2d()
        .set_title(title, &[])
        .points(&vx, &vy, &[PointSymbol('O'), Color(gnuplot::RGBString("blue"))]) // Original data points
        .points(&pos_x, &pos_y, &[PointSymbol('.'), Color(gnuplot::RGBString("green"))]) // Positive classified points
        .points(&neg_x, &neg_y, &[PointSymbol('.'), Color(gnuplot::RGBString("red"))]); // Negative classified points
                                                                    //.lines(&x_vals, &y_vals, &[Color("black")]);
    fg.show().unwrap();
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
