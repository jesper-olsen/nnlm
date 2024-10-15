use gnuplot::{AxesCommon, Figure};
use nalgebra::{DMatrix, DVector};
use nnlm::*;
use std::fmt;

#[derive(Copy, Clone)]
struct GKernal<const IDIM: usize> {
    pub mean: [f64; IDIM],
    pub var: [f64; IDIM],
}

impl<const IDIM: usize> fmt::Display for GKernal<IDIM> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "GKernal")?;

        write!(f, "mean: ");
        for j in 0..IDIM {
            write!(f, "{:>8.2} ", self.mean[j])?;
        }
        writeln!(f);
        write!(f, "var:  ");
        for j in 0..IDIM {
            write!(f, "{:>8.2} ", self.var[j])?;
        }
        writeln!(f)?;
        Ok(())
    }
}

impl<const IDIM: usize> GKernal<IDIM> {
    pub fn new() -> Self {
        Self {
            mean: [0.0; IDIM],
            var: [1.0; IDIM],
        }
    }

    pub fn estimate<'a, I>(&mut self, data: I)
    where
        I: IntoIterator<Item = &'a [f64; IDIM]>,
    {

            // Estimate mean and variance - Welford's method
            let mut count=0;
            self.mean.fill(0.0);
            self.var.fill(0.0);
            let mut old_mean = [0.0f64; IDIM];
            for (i, item) in data.into_iter().enumerate() {
                let v = item.as_ref();
                count+=1;
                old_mean.copy_from_slice(&self.mean);
                self.mean
                    .iter_mut()
                    .zip(v.iter())
                    .for_each(|(m, x)| *m += (*x - *m) / (i+1) as f64);
                for j in 0..IDIM {
                    self.var[j] += (v[j] - self.mean[j]) * (v[j] - old_mean[j]);
                }
            }
            self.var.iter_mut().for_each(|e| *e /= (count - 1) as f64);
    }

    /// Squared Mahalanobis distance between x and the Gaussian kernel.
    pub fn dist(&self, x: &[f64]) -> f64 {
        assert_eq!(x.len(), IDIM);
        self.mean
            .iter()
            .zip(x.iter())
            .zip(self.var.iter())
            .map(|((&m, &xi), &v)| (xi - m).powi(2) / v)
            .sum()
    }

    /// Log probability of x 
    pub fn logp(&self, x: &[f64]) -> f64 {
        assert_eq!(x.len(), IDIM);
        let s: f64 = self
            .var
            .iter()
            .map(|v| (2.0 * std::f64::consts::PI * v).ln())
            .sum();
        -0.5 * (self.dist(x) + s)
    }

    /// Probability of x 
    pub fn p(&self, x: &[f64]) -> f64 {
        assert_eq!(x.len(), IDIM);
        let a: f64 = (2.0 * std::f64::consts::PI).powi(IDIM as i32);
        let vproduct = self.var.iter().fold(a, |acc, v| acc * v);
        (1.0 / vproduct.sqrt()) * (-0.5 * self.dist(x)).exp()
    }
}

pub struct RBF<const IDIM: usize, const NKERNALS: usize> {
    pub kernal: [GKernal<IDIM>; NKERNALS],
    pub weights: [f64; NKERNALS],
}

impl<const IDIM: usize, const NKERNALS: usize> fmt::Display for RBF<IDIM, NKERNALS> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "RBF")?;

        for i in 0..NKERNALS {
            for j in 0..IDIM {
                write!(f, "{:>8.2} ", self.kernal[i])?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

fn rbf(dist: f64) {
    let central_radius = 10.0;
    let radius_variation = 6.0;
    let (data, labels) = halfmoons::<3000>(central_radius, radius_variation, dist);
    let mut model = RBF::<2, 20>::new();
    model.kmeans(&data, 100);
    println!("{model}");
}

impl<const IDIM: usize, const NKERNALS: usize> RBF<IDIM, NKERNALS> {
    pub fn new() -> Self {
        Self {
            kernal: [GKernal::new(); NKERNALS],
            weights: [0.0; NKERNALS],
        }
    }

    pub fn output(&self, x: &[f64]) -> f64 {
        assert_eq!(x.len(), IDIM, "Input length does not match expected IDIM");
        let mut result = 0.0;
        for i in 0..NKERNALS {
            result += self.weights[i] * self.kernal[i].p(x);
        }
        result
    }

    fn kmeans(&mut self, data: &[[f64; IDIM]], max_iter: usize) {

            let dta: Vec<&[f64;IDIM]> = data.iter()
                           .enumerate()
                           .filter(|(i,_)| true)
                           .map(|(_,v)| v).collect();

        let dta: Vec<&[f64;IDIM]> = data.iter().map(|v| v).collect();

        let mut k = GKernal::<IDIM>::new(); 
        k.estimate(data.iter().map(|v| v).collect::<Vec<_>>()); 
        println!("trained {k}");

        const EPSILON: f64 = 0.0;
        assert!(
            data.len() >= NKERNALS,
            "Not enough data samples to initialize centroids."
        );

        // Initialize kernals from the first NKERNALS data samples (assume randomised)
        for (k, sample) in self.kernal.iter_mut().zip(data.iter().take(NKERNALS)) {
            k.mean.copy_from_slice(sample);
            k.var.fill(1.0);
        }

        let mut new_kernal = [GKernal::new(); NKERNALS];
        let mut sample2cluster: Vec<usize> = Vec::with_capacity(data.len());

        for ep in 0..max_iter {
            sample2cluster.clear();
            let mut gdist = 0.0;

            // assign samples to their nearest cluster
            let mut cluster_counts: [usize; NKERNALS] = [0; NKERNALS];
            for x in data {
                let (min_dist, min_pos) = self.nearest_kernal(x);
                sample2cluster.push(min_pos);
                cluster_counts[min_pos]+=1;
                gdist += min_dist;
            }
            gdist /= data.len() as f64;

            // re-estimate kernals 
            for c in 0..NKERNALS {
                if cluster_counts[c]>1 {
                    let dta: Vec<&[f64; IDIM]> = data.iter()
                        .enumerate()
                        .filter(|(i, _)| sample2cluster[*i] == c)
                        .map(|(_, v)| v)
                        .collect();
                    new_kernal[c].estimate(dta);
                }
            }

            // check for convergence - distance betwen old and new kernal estimates
            let cdist: f64 = (0..NKERNALS)
                .map(|c| self.kernal[c].dist(&new_kernal[c].mean))
                .sum();
            println!("Kmeans ep: {ep}; gdist: {gdist:>5.2}; cdist: {cdist:>5.2}");

            std::mem::swap(&mut self.kernal, &mut new_kernal);

            if cdist <= EPSILON {
                break;
            }
        }
    }

    fn nearest_kernal(&self, x: &[f64; IDIM]) -> (f64, usize) {
        self.kernal
            .iter()
            .enumerate()
            .map(|(i, k)| (k.dist(x), i))
            .fold((f64::INFINITY, 0), |(min_dist, min_pos), (dist, i)| {
                if dist < min_dist {
                    (dist, i)
                } else {
                    (min_dist, min_pos)
                }
            })
    }
}

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

// least squares with L2 regularisation  (mu)
fn least_squares(dist: f64, mu: f64) {
    let central_radius = 10.0;
    let radius_variation = 6.0;
    let (data, labels) = halfmoons::<3000>(central_radius, radius_variation, dist);
    let mut data: Vec<_> = data
        .into_iter()
        .map(|[x, y]| [1.0, x, y]) // Add 1.0 - gives the perceptron a bias term
        .collect();

    const NTRAIN: usize = 1000;

    let mean = calc_mean(&data[..NTRAIN]);
    normalise_mean(&mut data[..NTRAIN], &mean);
    normalise_mean(&mut data[NTRAIN..], &mean);

    let max_vals = calc_max_vals(&data[..NTRAIN]);
    normalise_max_val(&mut data[..NTRAIN], &max_vals);
    normalise_max_val(&mut data[NTRAIN..], &max_vals);

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
    rbf(0.0);
    //for dist in [0.0] {
    //    for mu in [0.0, 0.1] {
    //        least_squares(dist, mu);
    //    }
    //}
}
