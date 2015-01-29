// Copyright ©2014 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dist

import (
	"math"
	"testing"

	"github.com/gonum/floats"
)

func TestExponentialProb(t *testing.T) {
	pts := []univariateProbPoint{
		univariateProbPoint{
			loc:     0,
			prob:    1,
			cumProb: 0,
			logProb: 0,
		},
		univariateProbPoint{
			loc:     -1,
			prob:    0,
			cumProb: 0,
			logProb: math.Inf(-1),
		},
		univariateProbPoint{
			loc:     1,
			prob:    1 / (math.E),
			cumProb: 0.6321205588285576784044762298385391325541888689682321654921631983025385042551001966428527256540803563,
			logProb: -1,
		},
		univariateProbPoint{
			loc:     20,
			prob:    math.Exp(-20),
			cumProb: 0.999999997938846377561442172034059619844179023624192724400896307027755338370835976215440646720089072,
			logProb: -20,
		},
	}
	testDistributionProbs(t, Exponential{Rate: 1}, "Exponential", pts)
}

func TestExponentialFitPrior(t *testing.T) {
	for i, test := range []struct {
		rate    float64
		samps   []float64
		weights []float64
	}{
		{
			rate:    13.7,
			samps:   randn(&Exponential{Rate: 13}, 10),
			weights: nil,
		},
		{
			rate:    13.7,
			samps:   randn(&Exponential{Rate: 13}, 10),
			weights: ones(10),
		},
		{
			rate:    13.7,
			samps:   randn(&Exponential{Rate: 13}, 10),
			weights: randn(&Exponential{Rate: 13}, 10),
		},
	} {
		// ensure that conjugate produces the same result both incrementally and all at once
		incDist := &Exponential{
			Rate: test.rate,
		}
		stats := make([]float64, incDist.NumSuffStat())
		prior := make([]float64, 1)
		if test.weights != nil {
			for j := range test.samps {
				nsInc := incDist.SuffStat(test.samps[j:j+1], test.weights[j:j+1], stats)
				incDist.ConjugateUpdate(stats, nsInc, prior)

				allDist := &Exponential{
					Rate: test.rate,
				}
				nsAll := allDist.SuffStat(test.samps[0:j+1], test.weights[0:j+1], stats)
				allDist.ConjugateUpdate(stats, nsAll, []float64{0})
				if !floats.EqualWithinAbs(incDist.Rate, allDist.Rate, 1e-14) {
					t.Errorf("prior doesn't match after incremental update for (%d, %d). Incremental is %v, all at once is %v", i, j, incDist.Rate, allDist.Rate)
				}
			}
		} else {
			for j := range test.samps {
				nsInc := incDist.SuffStat(test.samps[j:j+1], nil, stats)
				incDist.ConjugateUpdate(stats, nsInc, prior)

				allDist := &Exponential{
					Rate: test.rate,
				}
				nsAll := allDist.SuffStat(test.samps[0:j+1], nil, stats)
				allDist.ConjugateUpdate(stats, nsAll, []float64{0})
				if !floats.EqualWithinAbs(incDist.Rate, allDist.Rate, 1e-14) {
					t.Errorf("prior doesn't match after incremental update for (%d, %d). Incremental is %v, all at once is %v", i, j, incDist.Rate, allDist.Rate)
				}
			}
		}
	}
}
