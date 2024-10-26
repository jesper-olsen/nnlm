use na::DMatrix;
use nalgebra as na;
use std::fmt;
use stmc_rs::marsaglia::Marsaglia;

fn clip_gradient(grad: f64, max_grad: f64) -> f64 {
    if grad > max_grad {
        max_grad
    } else if grad < -max_grad {
        -max_grad
    } else {
        grad
    }
}

#[derive(Copy, Clone)]
pub struct GKernel<const IDIM: usize> {
    pub mean: [f64; IDIM],
    pub var: [f64; IDIM],
}

impl<const IDIM: usize> fmt::Display for GKernel<IDIM> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "GKernel")?;
        write!(f, "mean: ")?;
        for j in 0..IDIM {
            write!(f, "{:>8.2} ", self.mean[j])?;
        }
        writeln!(f)?;
        write!(f, "var:  ")?;
        for j in 0..IDIM {
            write!(f, "{:>8.2} ", self.var[j])?;
        }
        writeln!(f)?;
        Ok(())
    }
}

impl<const IDIM: usize> GKernel<IDIM> {
    pub fn new() -> Self {
        Self {
            mean: [0.0; IDIM],
            var: [1.0; IDIM],
        }
    }

    pub fn reset(&mut self, mean: &[f64], var: &[f64]) {
        self.mean.copy_from_slice(mean);
        self.var.copy_from_slice(var);
    }

    pub fn split(&mut self, other: &GKernel<IDIM>) {
        self.mean
            .iter_mut()
            .zip(other.mean.iter())
            .for_each(|(m, om)| *m = *om + 0.001);
        self.var.copy_from_slice(&other.var);
    }

    pub fn estimate_welford(&mut self, data: &[[f64; IDIM]]) {
        // Estimate mean and variance - Welford's method
        self.mean.fill(0.0);
        self.var.fill(0.0);
        let mut old_mean = [0.0f64; IDIM];
        for (i, v) in data.iter().enumerate() {
            old_mean.copy_from_slice(&self.mean);
            self.mean
                .iter_mut()
                .zip(v.iter())
                .for_each(|(m, x)| *m += (*x - *m) / (i + 1) as f64);
            for j in 0..IDIM {
                self.var[j] += (v[j] - self.mean[j]) * (v[j] - old_mean[j]);
            }
        }
        self.var
            .iter_mut()
            .for_each(|e| *e /= (data.len() - 1) as f64);
    }

    pub fn estimate(&mut self, data: &[[f64; IDIM]]) {
        // Estimate mean and variance - 2 pass
        self.mean.fill(0.0);
        self.var.fill(0.0);

        let n = data.len() as f64;
        for v in data {
            self.mean
                .iter_mut()
                .zip(v.iter())
                .for_each(|(m, x)| *m += *x);
        }
        self.mean.iter_mut().for_each(|m| *m /= n);

        for v in data {
            for i in 0..IDIM {
                self.var[i] += (v[i] - self.mean[i]).powi(2);
            }
        }
        self.var.iter_mut().for_each(|v| *v /= n - 1.0);
    }

    /// Squared Euclidian distance between x and the Gaussian kernel.
    pub fn dist_euc(&self, x: &[f64]) -> f64 {
        debug_assert_eq!(x.len(), IDIM);
        self.mean
            .iter()
            .zip(x.iter())
            .map(|(&m, &xi)| (xi - m).powi(2))
            .sum()
    }

    /// Squared Mahalanobis distance between x and the Gaussian kernel.
    pub fn dist(&self, x: &[f64]) -> f64 {
        debug_assert_eq!(x.len(), IDIM);
        self.mean
            .iter()
            .zip(x.iter())
            .zip(self.var.iter())
            .map(|((&m, &xi), &v)| (xi - m).powi(2) / v)
            .sum()
    }

    /// Log probability of x
    pub fn logp(&self, x: &[f64]) -> f64 {
        debug_assert_eq!(x.len(), IDIM);
        let s: f64 = self
            .var
            .iter()
            .map(|v| (2.0 * std::f64::consts::PI * v).ln())
            .sum();
        -0.5 * (self.dist(x) + s)
    }

    /// Probability of x
    pub fn p(&self, x: &[f64]) -> f64 {
        debug_assert_eq!(x.len(), IDIM);
        let a: f64 = (2.0 * std::f64::consts::PI).powi(IDIM as i32);
        let vproduct = self.var.iter().fold(a, |acc, v| acc * v);
        (1.0 / vproduct.sqrt()) * (-0.5 * self.dist(x)).exp()
    }

    pub fn floor_variances(&mut self, floor: f64) {
        self.var.iter_mut().for_each(|v| *v = v.max(floor));
    }

    pub fn set_variances(&mut self, val: f64) {
        self.var.iter_mut().for_each(|v| *v = val);
    }
}

pub struct RBF<const IDIM: usize> {
    pub kernels: Vec<GKernel<IDIM>>,
    pub weights: Vec<f64>,
}

impl<const IDIM: usize> fmt::Display for RBF<IDIM> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "RBF")?;

        write!(f, "Weights: ")?;
        for w in &self.weights {
            write!(f, "{:>8.2} ", w)?;
        }
        writeln!(f)?;

        for (i, k) in self.kernels.iter().enumerate() {
            write!(f, "{i}: {}", k)?;
        }
        Ok(())
    }
}

impl<const IDIM: usize> RBF<IDIM> {
    pub fn new(nkernels: usize) -> Self {
        Self {
            kernels: vec![GKernel::new(); nkernels],
            weights: vec![0.0f64; nkernels],
        }
    }

    pub fn output(&self, x: &[f64]) -> f64 {
        debug_assert_eq!(x.len(), IDIM, "Input length does not match expected IDIM");
        self.kernels
            .iter()
            .zip(self.weights.iter())
            .map(|(&k, &w)| w * k.p(x))
            .sum::<f64>()
            .signum()
    }

    pub fn eval(&self, data: &[[f64; IDIM]], labels: &[i8], title: &str) {
        let mut errors = 0;
        for (x, label) in data.iter().zip(labels) {
            let y = self.output(x) as i8;
            let d = label.signum();
            if d != y {
                errors += 1
            }
        }
        let errp = 100.0 * errors as f64 / data.len() as f64;
        println!("{title}: {errors}/{} = {errp:>6.2}%", data.len());
    }

    pub fn train_weights_rhs(
        &mut self,
        data: &[[f64; IDIM]],
        labels: &[i8],
        max_iter: usize,
    ) -> Vec<f64> {
        let sig = 0.01;
        let lambda = 1.0;
        let nkernels = self.kernels.len();
        let mut p = DMatrix::identity(nkernels, nkernels) / sig;

        let mut lmse = Vec::<f64>::new();
        for _ep in 0..max_iter {
            let mut mse = 0.0;
            for (_, (x, label)) in data.iter().zip(labels).enumerate() {
                let d = *label as f64; // 1 or -1
                let k: Vec<f64> = self.kernels.iter().map(|&k| k.p(x)).collect();

                let k_matrix = DMatrix::from_vec(k.len(), 1, k.clone());

                let pai = &p * &k_matrix;
                let g_t_pai = k
                    .iter()
                    .zip(pai.iter())
                    .map(|(gi, pi)| gi * pi)
                    .sum::<f64>();
                let kk = pai / (lambda + g_t_pai);

                let e = d - self
                    .weights
                    .iter()
                    .zip(&k)
                    .map(|(wi, gi)| wi * gi)
                    .sum::<f64>();
                let w_delta: Vec<f64> = kk.iter().map(|&kki| kki * e).collect();
                mse += e * e;

                for (wi, wd) in self.weights.iter_mut().zip(w_delta.iter()) {
                    *wi += wd;
                }

                let g_t_p = k_matrix.transpose() * &p;
                let kk_g_t_p = kk * g_t_p;
                p = (&p - kk_g_t_p) / lambda;
            }
            mse /= data.len() as f64;
            lmse.push(mse);
        }
        lmse
    }

    pub fn train_weights_lms(
        &mut self,
        data: &[[f64; IDIM]],
        labels: &[i8],
        max_iter: usize,
    ) -> Vec<f64> {
        let mut lmse = Vec::<f64>::new();
        for ep in 0..max_iter {
            let lr = 0.1 / (1.0 + ep as f64);
            let mut mse = 0.0;
            for (x, label) in data.iter().zip(labels) {
                let k: Vec<f64> = self.kernels.iter().map(|&k| k.p(x)).collect();
                let d = *label as f64;
                let y: f64 = k
                    .iter()
                    .zip(self.weights.iter())
                    .map(|(g, w)| g * w)
                    .sum::<f64>();
                let e = d - y;

                mse += e * e;
                self.weights
                    .iter_mut()
                    .zip(k.iter())
                    .map(|(w, ki)| (w, lr * e * ki))
                    .for_each(|(w, grad)| *w += grad);
                self.normalise_weights();
            }
            mse /= data.len() as f64;
            lmse.push(mse);
            //println!("ep {ep} {mse}");
        }
        lmse
    }

    fn normalise_weights(&mut self) {
        let nw = self.weights.iter().map(|w| w * w).sum::<f64>().sqrt();
        if nw > 0.0 {
            self.weights.iter_mut().for_each(|w| *w /= nw);
        }
    }

    pub fn train_kernels_em(&mut self, rng: &mut Marsaglia, data: &[[f64; IDIM]], max_iter: usize) {
        const EPSILON: f64 = 0.0;
        debug_assert!(
            data.len() >= self.kernels.len(),
            "Not enough data samples to initialize kernels."
        );

        self.train_kernels_kmeans(rng, data, max_iter);
        let mut new_kernels = vec![GKernel::<IDIM>::new(); self.kernels.len()];
        let mut sample2gamma: Vec<Vec<f64>> = Vec::with_capacity(data.len());

        for ep in 0..max_iter {
            sample2gamma.clear();
            // E-step
            for x in data {
                let mut gamma: Vec<f64> = self
                    .weights
                    .iter()
                    .zip(self.kernels.iter())
                    .map(|(w, k)| w * k.p(x))
                    .collect();
                let gsum: f64 = gamma.iter().sum();
                if gsum > 0.0 {
                    gamma.iter_mut().for_each(|g| *g /= gsum);
                }
                sample2gamma.push(gamma);
            }

            // M step - re-estimate kernels
            let gksum =
                sample2gamma
                    .iter()
                    .fold(vec![0.0f64; self.kernels.len()], |mut acc, gamma| {
                        acc.iter_mut().zip(gamma.iter()).for_each(|(s, &g)| *s += g);
                        acc
                    });

            new_kernels
                .iter_mut()
                .enumerate()
                .filter(|(k, _)| gksum[*k] > 10.0)
                .for_each(|(k, knl)| {
                    for (x, gamma) in data.iter().zip(sample2gamma.iter()) {
                        knl.mean
                            .iter_mut()
                            .zip(x.iter())
                            .for_each(|(m, e)| *m += gamma[k] * e);
                    }
                    knl.mean.iter_mut().for_each(|m| *m /= gksum[k]);

                    for (x, gamma) in data.iter().zip(sample2gamma.iter()) {
                        knl.var
                            .iter_mut()
                            .zip(knl.mean.iter())
                            .zip(x.iter())
                            .for_each(|((v, m), e)| *v += gamma[k] * (m - e).powi(2));
                    }
                    knl.var.iter_mut().for_each(|v| *v /= gksum[k]);
                    self.weights[k] = gksum[k] / data.len() as f64;
                });

            let defunkt_kernels: Vec<usize> = self
                .weights
                .iter()
                .enumerate()
                .filter_map(|(i, &weight)| if weight < 0.01 { Some(i) } else { None })
                .rev()
                .collect();
            for c in defunkt_kernels {
                println!("defunkt kernel {c} - removing");
                self.weights.remove(c);
                self.kernels.remove(c);
                new_kernels.remove(c);
            }

            // check for convergence - distance betwen old and new kernel estimates
            let cdist: f64 = self
                .kernels
                .iter()
                .zip(new_kernels.iter())
                .map(|(k, nk)| k.dist_euc(&nk.mean))
                .sum();
            println!("EM training - ep: {ep}; cdist: {cdist:>5.2}");
            if cdist <= EPSILON {
                break;
            }
        }
        println!("EM model: {self}");
    }

    pub fn train_kernels_kmeans(
        &mut self,
        rng: &mut Marsaglia,
        data: &[[f64; IDIM]],
        max_iter: usize,
    ) -> f64 {
        const EPSILON: f64 = 0.0;
        debug_assert!(
            data.len() >= self.kernels.len(),
            "Not enough data samples to initialize centroids."
        );

        // Initialize kernel means from the first nkernels data samples (assume randomised)
        let mut global_kernel = GKernel::<IDIM>::new();
        global_kernel.estimate_welford(data); // please the borrow checker
        for k in &mut self.kernels {
            k.reset(
                &data[(rng.uni() * data.len() as f64) as usize],
                &global_kernel.var,
            );
        }

        let mut new_kernels = vec![GKernel::new(); self.kernels.len()];
        let mut sample2kernel: Vec<usize> = Vec::with_capacity(data.len());

        let mut gdist = 0.0f64;
        for ep in 0..max_iter {
            sample2kernel.clear();
            // E-step - assign samples to their nearest cluster
            gdist = data
                .iter()
                .map(|x| self.nearest_kernel(x))
                .map(|(dist, k)| {
                    sample2kernel.push(k);
                    dist
                })
                .sum();

            // M step - re-estimate kernels
            let kcounts: Vec<usize> =
                sample2kernel
                    .iter()
                    .fold(vec![0; self.kernels.len()], |mut acc, &k| {
                        acc[k] += 1;
                        acc
                    });
            self.weights
                .iter_mut()
                .zip(kcounts.iter())
                .for_each(|(w, cnt)| *w = *cnt as f64 / sample2kernel.len() as f64);

            for (c, k) in new_kernels.iter_mut().enumerate() {
                if kcounts[c] < 5 {
                    println!(
                        "Warning: kernel {c} only has {} samples - resetting",
                        kcounts[c]
                    );
                    k.reset(
                        &data[(rng.uni() * data.len() as f64) as usize],
                        &global_kernel.var,
                    );
                } else {
                    // Estimate mean and variance - Welford's method
                    k.mean.fill(0.0);
                    k.var.fill(0.0);
                    let mut old_mean = [0.0f64; IDIM];
                    data.iter()
                        .zip(sample2kernel.iter())
                        .filter(|(_, z)| **z == c)
                        .map(|(v, _)| v)
                        .enumerate()
                        .for_each(|(i, v)| {
                            old_mean.copy_from_slice(&k.mean);
                            k.mean
                                .iter_mut()
                                .zip(v.iter())
                                .for_each(|(m, x)| *m += (*x - *m) / (i + 1) as f64);
                            for j in 0..IDIM {
                                k.var[j] += (v[j] - k.mean[j]) * (v[j] - old_mean[j]);
                            }
                        });
                    k.var.iter_mut().for_each(|e| *e /= (kcounts[c] - 1) as f64);
                }
            }

            std::mem::swap(&mut self.kernels, &mut new_kernels);

            // check for convergence - distance betwen old and new kernel estimates
            let cdist: f64 = self
                .kernels
                .iter()
                .zip(new_kernels.iter())
                .map(|(k, nk)| k.dist_euc(&nk.mean))
                .sum();

            println!("Kmeans ep: {ep}; gdist: {gdist:.2}; cdist: {cdist:>5.2}");
            if cdist <= EPSILON {
                break;
            }
        }
        gdist
    }

    fn nearest_kernel(&self, x: &[f64; IDIM]) -> (f64, usize) {
        self.kernels
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
