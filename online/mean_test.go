// Copyright ©2014 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package incremental

import (
	"math"
	"testing"
)

func Panics(fun func()) (b bool) {
	defer func() {
		err := recover()
		if err != nil {
			b = true
		}
	}()
	fun()
	return
}

func TestMeanIncrement(t *testing.T) {
	for i, test := range []struct {
		x   []float64
		ans []float64
	}{
		{
			x:   []float64{8, -3, 7, 8, 10},
			ans: []float64{8, 2.5, 4, 5, 6},
		},
	} {
		m1 := NewMean()
		for j, v := range test.x {
			m1.Increment(v)
			if mu := m1.Result(); math.Abs(test.ans[j]-mu) > 1e-14 {
				t.Errorf("Increment mean mismatch case %d, %d. Expected %v, Found %v", i, j, test.ans[j], mu)
			}
			m2 := NewMean()
			m2.M = m1.M
			m2.N = m1.N

			m2.IncrementSlice(test.x[j+1:])
			if mu := m2.Result(); math.Abs(test.ans[len(test.ans)-1]-mu) > 1e-14 {
				t.Errorf("IncrementalSlice mean mismatch case %d, %d. Expected %v, Found %v", i, j, test.ans[len(test.ans)-1], mu)
			}
		}
	}
}

func TestMeanIncrementWeighted(t *testing.T) {
	for i, test := range []struct {
		x   []float64
		wts []float64
		ans []float64
	}{
		{
			x:   []float64{8, -3, 7, 8, 10},
			wts: []float64{1, 0, 1, -1, 1},
			ans: []float64{8, 8, 7.5, 7, 8.5},
		},
	} {
		m := NewMean()
		for j, v := range test.x {
			m.IncrementWeighted(v, test.wts[j])
			if mu := m.Result(); math.Abs(test.ans[j]-mu) > 1e-14 {
				t.Errorf("IncrementWeighted mean mismatch case %d, %d. Expected %v, Found %v", i, j, test.ans[j], mu)
			}

			m2 := NewMean()
			m2.M = m.M
			m2.N = m.N

			m2.IncrementSliceWeighted(test.x[j+1:], test.wts[j+1:])
			if mu := m2.Result(); math.Abs(test.ans[len(test.ans)-1]-mu) > 1e-14 {
				t.Errorf("IncrementalSliceWeighted mean mismatch case %d, %d. Expected %v, Found %v", i, j, test.ans[len(test.ans)-1], mu)
			}
		}
	}
	if !Panics(func() { NewMean().IncrementSliceWeighted(make([]float64, 2), make([]float64, 3)) }) {
		t.Errorf("IncrementSliceWeighted did not panic with x, length length mismatch")
	}

}

func TestMeanCombine(t *testing.T) {
	for i, test := range []struct {
		x1, x2 []float64
		ans    float64
	}{
		{
			x1:  []float64{4, -3, 7, 8, 10},
			x2:  []float64{1, 2, 3, 4},
			ans: 4},
		{
			ans: 0,
		},
	} {
		m1 := NewMean(test.x1...)
		m2 := NewMean(test.x2...)
		m1.Combine(m2)
		if mu := m1.Result(); math.Abs(test.ans-mu) > 1e-14 {
			t.Errorf("Combined mean mismatch case %d. Expected %v, Found %v", i, test.ans, mu)
		}
	}
}
