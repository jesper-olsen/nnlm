use rand::Rng;

/// Halfmoons: randomly samples two "half moons". Returns samples and their labels. Samples are 3D:
/// 1 (bias) + 2d coordinates 
pub fn halfmoons<const NSAMP: usize>(
    central_radius: f64,
    radius_variation: f64,
    dist: f64,
) -> ([[f64; 3]; NSAMP], [i8; NSAMP]) {
    // half doughnut
    // radius_variation: diameter of doughnut ring
    // central_radius: center radius of dounut - sampels are at rad += radius_variation/2

    if central_radius < radius_variation / 2.0 {
        panic!("The central_radius should be at least larger than half the radius_variation");
    }

    let mut rng = rand::thread_rng();
    let mut data = [[0.0; 3]; NSAMP];
    let mut labels = [0; NSAMP];
    for i in (0..NSAMP).step_by(2) {
        let radius = central_radius - radius_variation / 2.0 + radius_variation * rng.gen::<f64>();
        let theta = std::f64::consts::PI * rng.gen::<f64>();
        data[i] = [1.0, radius * theta.cos(), radius * theta.sin()];
        labels[i] = 1;

        let radius = central_radius - radius_variation / 2.0 + radius_variation * rng.gen::<f64>();
        let theta = std::f64::consts::PI * rng.gen::<f64>();
        data[i + 1] = [
            1.0,
            -radius * theta.cos() + central_radius,
            -radius * theta.sin() - dist,
        ];
        labels[i + 1] = -1;
    }

    (data, labels)
}

pub struct Perceptron<const N: usize> {
    pub weights: [f64; N],
}

impl<const N: usize> Perceptron<N> {
    pub fn new() -> Self {
        Self {
            weights: [0.0; N], // including bias
        }
    }

    pub fn train(&mut self, data: &[[f64; N]], labels: &[i8]) -> f64 {
        assert_eq!(data.len(), labels.len());

        let mut ee = 0.0;
        for (x, &d) in data.iter().zip(labels.iter()) {
            let y = sign(
                self.weights
                    .iter()
                    .zip(x.iter())
                    .map(|(wi, xi)| wi * xi)
                    .sum::<f64>(),
            );
            let error = (d - y) as f64;
            for (wj, xj) in self.weights.iter_mut().zip(x.iter()) {
                let lr = 1.0; // Optionally impl annealing, e.g. linear decay
                *wj += lr * error * xj;
            }
            ee += error * error;
        }
        ee /= data.len() as f64;
        ee
    }

    pub fn classify(&self, x: &[f64]) -> i8 {
        sign(
            self.weights
                .iter()
                .zip(x.iter())
                .map(|(wi, xi)| wi * xi)
                .sum::<f64>(),
        )
    }
}

pub fn sign(value: f64) -> i8 {
    if value >= 0.0 {
        1
    } else {
        -1
    }
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
