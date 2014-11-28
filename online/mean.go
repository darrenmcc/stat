// Copyright ©2014 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package incremental

// Adapted from Pébay, Philippe. "Formulas for robust, one-pass parallel
// computation of covariances and arbitrary-order statistical moments."
// Sandia Report SAND2008-6212, Sandia National Laboratories (2008)

type mean struct {
	M float64
	N float64
}

// NewMean creates a new incremental mean measurement.
func NewMean(x ...float64) *mean {
	m := new(mean)
	m.IncrementSlice(x)
	return m
}

// Increment adds a new sample to take a mean over.
func (m *mean) Increment(y float64) {
	m.N++
	m.M += (y - m.M) / m.N
}

// IncrementWeighted adds a new sample to take a mean over, using a given weight.
func (m *mean) IncrementWeighted(y, weight float64) {
	if weight == 0.0 {
		return
	}
	m.N += weight
	m.M += (y - m.M) * weight / m.N
}

// Result returns the current mean Result.
func (m *mean) Result() float64 {
	return m.M
}

// Combine two Means into a single Mean.
func Combine(m1, m2 *mean) *mean {
	if m1.N == 0 && m2.N == 0 {
		return NewMean()
	}
	return &mean{
		M: m1.M + m2.N/(m1.N+m2.N)*(m2.M-m1.M),
		N: m1.N + m2.N,
	}
}

// IncrementSlice updates a Mean with a slice of inputs.
func (m *mean) IncrementSlice(x []float64) {
	var mu, N float64
	N = float64(len(x))
	if N == 0 {
		return
	}
	for _, y := range x {
		mu += y / N
	}
	m.N += N
	m.M += N / m.N * (mu - m.M)
}

// IncrementSliceWeighted updates a Mean with a weighted slice of inputs.
func (m *mean) IncrementSliceWeighted(x, weights []float64) {
	if len(x) != len(weights) {
		panic("stat: slice length mismatch")
	}

	// this uses a pairwise approach, which should increase numerical stability.
	var mu, N float64
	for i, weight := range weights {
		if weight == 0.0 {
			continue
		}
		N += weight
		mu += (x[i] - mu) * weight / N
	}
	m.N += N
	m.M += N / m.N * (mu - m.M)

}
