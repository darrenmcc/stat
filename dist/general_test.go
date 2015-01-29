// Copyright ©2014 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dist

import (
	"fmt"
	"math"
	"testing"
)

type univariateProbPoint struct {
	loc     float64
	logProb float64
	cumProb float64
	prob    float64
}

type UniProbDist interface {
	Prob(float64) float64
	CDF(float64) float64
	LogProb(float64) float64
	Quantile(float64) float64
	Survival(float64) float64
}

func absEq(a, b float64) bool {
	if math.Abs(a-b) > 1e-14 {
		return false
	}
	return true
}

// TODO: Implement a better test for Quantile
func testDistributionProbs(t *testing.T, dist UniProbDist, name string, pts []univariateProbPoint) {
	for _, pt := range pts {
		logProb := dist.LogProb(pt.loc)
		if !absEq(logProb, pt.logProb) {
			t.Errorf("Log probability doesnt match for "+name+". Expected %v. Found %v", pt.logProb, logProb)
		}
		prob := dist.Prob(pt.loc)
		if !absEq(prob, pt.prob) {
			t.Errorf("Probability doesn't match for "+name+". Expected %v. Found %v", pt.prob, prob)
		}
		cumProb := dist.CDF(pt.loc)
		if !absEq(cumProb, pt.cumProb) {
			t.Errorf("Cumulative Probability doesn't match for "+name+". Expected %v. Found %v", pt.cumProb, cumProb)
		}
		if !absEq(dist.Survival(pt.loc), 1-pt.cumProb) {
			t.Errorf("Survival doesn't match for %v. Expected %v, Found %v", name, 1-pt.cumProb, dist.Survival(pt.loc))
		}
		if pt.prob != 0 {
			if math.Abs(dist.Quantile(pt.cumProb)-pt.loc) > 1e-4 {
				fmt.Println("true =", pt.loc)
				fmt.Println("calculated=", dist.Quantile(pt.cumProb))
				t.Errorf("Quantile doesn't match for "+name+", loc =  %v", pt.loc)
			}
		}
	}
}
