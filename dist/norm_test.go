// Copyright ©2014 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dist

import (
	"math"
	"testing"

	"github.com/gonum/floats"
)

// TestNormalProbs tests LogProb, Prob, CumProb, and Quantile
func TestNormalProbs(t *testing.T) {
	pts := []univariateProbPoint{
		univariateProbPoint{
			loc:     0,
			prob:    oneOverRoot2Pi,
			cumProb: 0.5,
			logProb: -0.91893853320467274178032973640561763986139747363778341281715,
		},
		univariateProbPoint{
			loc:     -1,
			prob:    0.2419707245191433497978301929355606548286719707374350254875550842811000635700832945083112946939424047,
			cumProb: 0.158655253931457051414767454367962077522087033273395609012605,
			logProb: math.Log(0.2419707245191433497978301929355606548286719707374350254875550842811000635700832945083112946939424047),
		},
		univariateProbPoint{
			loc:     1,
			prob:    0.2419707245191433497978301929355606548286719707374350254875550842811000635700832945083112946939424047,
			cumProb: 0.841344746068542948585232545632037922477912966726604390987394,
			logProb: math.Log(0.2419707245191433497978301929355606548286719707374350254875550842811000635700832945083112946939424047),
		},
		univariateProbPoint{
			loc:     -7,
			prob:    9.134720408364593342868613916794233023000190834851937054490546361277622761970225469305158915808284566e-12,
			cumProb: 1.279812543885835004383623690780832998032844154198717929e-12,
			logProb: math.Log(9.134720408364593342868613916794233023000190834851937054490546361277622761970225469305158915808284566e-12),
		},
		univariateProbPoint{
			loc:     7,
			prob:    9.134720408364593342868613916794233023000190834851937054490546361277622761970225469305158915808284566e-12,
			cumProb: 0.99999999999872018745611416499561637630921916700196715584580,
			logProb: math.Log(9.134720408364593342868613916794233023000190834851937054490546361277622761970225469305158915808284566e-12),
		},
	}
	testDistributionProbs(t, Normal{Mu: 0, Sigma: 1}, "normal", pts)

	pts = []univariateProbPoint{
		univariateProbPoint{
			loc:     2,
			prob:    0.07978845608028653558798921198687637369517172623298693153318516593413158517986036770025046678146138729,
			cumProb: 0.5,
			logProb: math.Log(0.07978845608028653558798921198687637369517172623298693153318516593413158517986036770025046678146138729),
		},
		univariateProbPoint{
			loc:     -3,
			prob:    0.04839414490382866995956603858711213096573439414748700509751101685622001271401665890166225893878848095,
			cumProb: 0.158655253931457051414767454367962077522087033273395609012605,
			logProb: math.Log(0.04839414490382866995956603858711213096573439414748700509751101685622001271401665890166225893878848095),
		},
		univariateProbPoint{
			loc:     7,
			prob:    0.04839414490382866995956603858711213096573439414748700509751101685622001271401665890166225893878848095,
			cumProb: 0.841344746068542948585232545632037922477912966726604390987394,
			logProb: math.Log(0.04839414490382866995956603858711213096573439414748700509751101685622001271401665890166225893878848095),
		},
		univariateProbPoint{
			loc:     -33,
			prob:    1.826944081672918668573722783358846604600038166970387410898109272255524552394045093861031783161656913e-12,
			cumProb: 1.279812543885835004383623690780832998032844154198717929e-12,
			logProb: math.Log(1.826944081672918668573722783358846604600038166970387410898109272255524552394045093861031783161656913e-12),
		},
		univariateProbPoint{
			loc:     37,
			prob:    1.826944081672918668573722783358846604600038166970387410898109272255524552394045093861031783161656913e-12,
			cumProb: 0.99999999999872018745611416499561637630921916700196715584580,
			logProb: math.Log(1.826944081672918668573722783358846604600038166970387410898109272255524552394045093861031783161656913e-12),
		},
	}
	testDistributionProbs(t, Normal{Mu: 2, Sigma: 5}, "normal", pts)
}

func TestNormalFitPrior(t *testing.T) {
	for i, test := range []struct {
		mu, sigma float64
		samps     []float64
		weights   []float64
	}{
		{
			mu:      2,
			sigma:   5,
			samps:   randn(&Normal{Mu: 2, Sigma: 5}, 10),
			weights: nil,
		},
		{
			mu:      2,
			sigma:   5,
			samps:   randn(&Normal{Mu: 2, Sigma: 5}, 10),
			weights: ones(10),
		},
		{
			mu:      2,
			sigma:   5,
			samps:   randn(&Normal{Mu: 2, Sigma: 5}, 10),
			weights: randn(&Exponential{Rate: 1}, 10),
		},
	} {
		// ensure that conjugate produces the same result both incrementally and all at once
		incDist := &Normal{
			Mu:    test.mu,
			Sigma: test.sigma,
		}
		stats := make([]float64, incDist.NumSuffStat())
		prior := make([]float64, 2)
		if test.weights != nil {
			for j := range test.samps {
				nsInc := incDist.SuffStat(test.samps[j:j+1], test.weights[j:j+1], stats)
				incDist.ConjugateUpdate(stats, nsInc, prior)

				allDist := &Normal{
					Mu:    test.mu,
					Sigma: test.sigma,
				}
				nsAll := allDist.SuffStat(test.samps[0:j+1], test.weights[0:j+1], stats)
				allDist.ConjugateUpdate(stats, nsAll, []float64{0, 0})
				if !floats.EqualWithinAbs(incDist.Mu, allDist.Mu, 1e-14) {
					t.Errorf("prior doesn't match after incremental update for (%d, %d). Incremental Mu is %v, all at once Mu is %v", i, j, incDist.Mu, allDist.Mu)
				}
				if !floats.EqualWithinAbs(incDist.Sigma, allDist.Sigma, 1e-14) {
					t.Errorf("prior doesn't match after incremental update for (%d, %d). Incremental Sigma is %v, all at once Sigma is %v", i, j, incDist.Sigma, allDist.Sigma)
				}
			}
		} else {
			for j := range test.samps {
				nsInc := incDist.SuffStat(test.samps[j:j+1], nil, stats)
				incDist.ConjugateUpdate(stats, nsInc, prior)

				allDist := &Normal{
					Mu:    test.mu,
					Sigma: test.sigma,
				}
				nsAll := allDist.SuffStat(test.samps[0:j+1], nil, stats)
				allDist.ConjugateUpdate(stats, nsAll, []float64{0, 0})
				if !floats.EqualWithinAbs(incDist.Mu, allDist.Mu, 1e-14) {
					t.Errorf("prior doesn't match after incremental update for (%d, %d). Incremental Mu is %v, all at once Mu is %v", i, j, incDist.Mu, allDist.Mu)
				}
				if !floats.EqualWithinAbs(incDist.Sigma, allDist.Sigma, 1e-14) {
					t.Errorf("prior doesn't match after incremental update for (%d, %d). Incremental Sigma is %v, all at once Sigma is %v", i, j, incDist.Sigma, allDist.Sigma)
				}
			}
		}
	}
}
