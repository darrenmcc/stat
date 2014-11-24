// Copyright ©2014 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package incremental

import (
	"math"
	"testing"
)

func TestMean(t *testing.T) {
	for i, test := range []struct {
		x   []float64
		ans []float64
	}{
		{
			x:   []float64{8, -3, 7, 8, 10},
			ans: []float64{8, 2.5, 4, 5, 6},
		},
	} {
		m := NewMean()
		for j, v := range test.x {
			m.Increment(v)
			if mu := m.Result(); math.Abs(test.ans[j]-mu) > 1e-14 {
				t.Errorf("Incremental mean mismatch case %d. Expected %v, Found %v", i, test.ans[j], mu)
			}
		}
	}
}
