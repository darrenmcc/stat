// Copyright ©2014 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package incremental

type mean struct {
	Sum float64
	N   int
}

// NewMean creates a new incremental mean measurement.
func NewMean() *mean {
	return new(mean)
}

// Increment adds a new sample to take a mean over.
func (m *mean) Increment(v float64) {
	m.Sum += v
	m.N++
}

// Result returns the current mean Result.
func (m *mean) Result() float64 {
	return m.Sum / float64(m.N)
}

// should this have a combine, which adds two means together?
// I think it should, because it can give the building blocks for how
// to perform it concurrently
/*
func (m1 *mean) Combine(m2 *mean) *mean {
	return &mean{
		Sum: m1.Sum + m2.Sum
		N: m1.N + m2.N
	}
}
*/

// should this have an IncrementMany, which ranges over input slices?
// I think it shouldn't, because it is partially redundant to Increment, but
// it does avoid repeatedly incrementing the count N.
/*
func (m *mean) IncrementMany(x ...float64) *mean {
	for _, v := range x {
		m.Sum += v
	}
	m.N += len(x)
}
*/
