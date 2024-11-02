use crate::kernel::GKernel;
use log::{info, warn};
use std::fmt;
use stmc_rs::marsaglia::Marsaglia;

pub struct GMM<const IDIM: usize> {
    pub kernels: Vec<GKernel<IDIM>>,
    pub weights: Vec<f64>,
}

impl<const IDIM: usize> fmt::Display for GMM<IDIM> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "GMM")?;

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

impl<const IDIM: usize> GMM<IDIM> {
    pub fn new(nkernels: usize) -> Self {
        Self {
            kernels: vec![GKernel::new(); nkernels],
            weights: vec![1.0f64 / nkernels as f64; nkernels],
        }
    }

    pub fn output(&self, x: &[f64; IDIM]) -> f64 {
        self.kernels
            .iter()
            .zip(self.weights.iter())
            .map(|(&k, &w)| w * k.p(x))
            .sum::<f64>()
    }

    fn kmeans_step(
        &mut self,
        data: &[[f64; IDIM]],
        new_kernels: &mut Vec<GKernel<IDIM>>,
        sample2kernel: &mut Vec<usize>,
    ) -> f64 {
        sample2kernel.clear();

        // E-step - assign samples to their nearest cluster
        data.iter()
            .map(|x| self.nearest_kernel(x))
            .for_each(|(_, k)| sample2kernel.push(k));

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
            self.weights[c] = kcounts[c] as f64 / data.len() as f64;
            if kcounts[c] > 1 {
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
        self.remove_defunkt_kernels(new_kernels);
        self.check_convergence(new_kernels)
    }

    fn largest_mixture(&self) -> usize {
        let (i, _) = self
            .weights
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        i
    }

    pub fn train_kernels_hierarchical(
        &mut self,
        rng: &mut Marsaglia,
        data: &[[f64; IDIM]],
        max_iter: usize,
    ) {
        let nkernels = self.kernels.len();
        self.kernels.drain(1..nkernels);
        self.weights.drain(1..nkernels);
        self.kernels[0].estimate(data);
        self.weights[0] = 1.0;
        // split the kernel with the largest weight in each iteration
        // Limit number of iterations - because of defunkt mixtures
        for _ in 0..2 * nkernels {
            if self.kernels.len() >= nkernels {
                break;
            }
            let i = self.largest_mixture();
            let k = self.kernels[i].split(rng);
            self.kernels.push(k);
            self.weights[i] /= 2.0;
            self.weights.push(self.weights[i]);
            self.train_kernels_em(rng, data, max_iter);
        }
    }

    /// initialise with random samples
    // fn initialise_kernels_kmeans(&mut self, rng: &mut Marsaglia, data: &[[f64; IDIM]]) {
    //     let mut global_kernel = GKernel::<IDIM>::new();
    //     global_kernel.estimate(data);
    //     for k in &mut self.kernels {
    //         k.reset(
    //             &data[(rng.uni() * data.len() as f64) as usize],
    //             &global_kernel.var,
    //         );
    //     }
    // }

    /// initialise with dispearsed random samples (kmeans++)
    fn initialise_kernels_kmeanspp(&mut self, rng: &mut Marsaglia, data: &[[f64; IDIM]]) {
        let mut global_kernel = GKernel::<IDIM>::new();
        global_kernel.estimate(data);

        self.kernels[0].reset(
            &data[(rng.uni() * data.len() as f64) as usize],
            &global_kernel.var,
        );

        // from sample to to closest kernel
        let mut min_distances = Vec::with_capacity(data.len());

        for k in 1..self.kernels.len() {
            min_distances.clear();
            data.iter()
                .map(|x| self.nearest_kernel(x))
                .for_each(|(d, _)| min_distances.push(d));
            let target: f64 = rng.uni() * min_distances.iter().sum::<f64>();
            let mut cd = 0.0;
            for (i, &d) in min_distances.iter().enumerate() {
                cd += d;
                if cd >= target {
                    self.kernels[k].reset(&data[i], &global_kernel.var);
                    break;
                }
            }
        }
    }

    pub fn train_kernels_kmeans(
        &mut self,
        rng: &mut Marsaglia,
        data: &[[f64; IDIM]],
        max_iter: usize,
    ) {
        const EPSILON: f64 = 0.0;
        debug_assert!(
            data.len() >= self.kernels.len(),
            "Not enough data samples to initialize centroids."
        );

        //self.initialise_kernels_kmeans(rng, data);
        self.initialise_kernels_kmeanspp(rng, data);

        let mut new_kernels = vec![GKernel::new(); self.kernels.len()];
        let mut sample2kernel: Vec<usize> = Vec::with_capacity(data.len());
        for ep in 0..max_iter {
            let cdist = self.kmeans_step(data, &mut new_kernels, &mut sample2kernel);
            info!("Kmeans ep: {ep}; cdist: {cdist:>5.2}");
            if cdist <= EPSILON {
                break;
            }
        }
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

    /// Note - assumes GMM - weights are kmeans/EM trained likelihoods, not RBF weights
    fn remove_defunkt_kernels(&mut self, new_kernels: &mut Vec<GKernel<IDIM>>) {
        let defunkt_kernels: Vec<usize> = self
            .weights
            .iter()
            .enumerate()
            .filter_map(|(i, &weight)| if weight < 0.01 { Some(i) } else { None })
            .rev()
            .collect();
        for c in defunkt_kernels {
            warn!("defunkt kernel {c} - removing");
            self.weights.remove(c);
            self.kernels.remove(c);
            new_kernels.remove(c);
        }
        std::mem::swap(&mut self.kernels, new_kernels);
    }

    /// distance betwen old and new kernel estimates
    fn check_convergence(&self, new_kernels: &Vec<GKernel<IDIM>>) -> f64 {
        self.kernels
            .iter()
            .zip(new_kernels.iter())
            .map(|(k, nk)| k.dist_euc(&nk.mean))
            .sum()
    }

    fn em_step(
        &mut self,
        data: &[[f64; IDIM]],
        new_kernels: &mut Vec<GKernel<IDIM>>,
        sample2gamma: &mut Vec<Vec<f64>>,
    ) -> f64 {
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
        let gksum = sample2gamma
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

        self.remove_defunkt_kernels(new_kernels);
        self.check_convergence(new_kernels)
    }

    pub fn train_kernels_em(&mut self, rng: &mut Marsaglia, data: &[[f64; IDIM]], max_iter: usize) {
        const EPSILON: f64 = 0.0;
        debug_assert!(
            data.len() >= self.kernels.len(),
            "Not enough data samples to initialize kernels."
        );

        self.initialise_kernels_kmeanspp(rng, data);
        let mut new_kernels = vec![GKernel::<IDIM>::new(); self.kernels.len()];
        let mut sample2gamma: Vec<Vec<f64>> = Vec::with_capacity(data.len());

        for ep in 0..max_iter {
            let cdist = self.em_step(data, &mut new_kernels, &mut sample2gamma);

            info!("EM training - ep: {ep}; cdist: {cdist:>5.2}");
            if cdist <= EPSILON {
                break;
            }
        }
    }
}
