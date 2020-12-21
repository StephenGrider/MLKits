const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class LinearRegression {
  constructor(features, labels, options) {
    this.features = features;
    this.labels = labels;
    this.options = Object.assign(
      {
        learningRate: 0.1,
        iterations: 1000,
      },
      options
    );

    // by convention initial guesses are given the value of either 0 or 1
    this.m = 0;
    this.b = 0;
  }

  gradientDescent() {
    const currentGuessesForMPG = this.features.map((row) => {
      // calculate mx + b values with respect to MPG
      // row[0] is assumed to be MPG
      return this.m * row[0] + this.b;
    });

    // 2/n ∑ ((mx+b) - Actual)
    const bSlope =
      (_.sum(
        currentGuessesForMPG.map((guess, i) => {
          // subtract actual mpg from our guess
          // this.labels is an 2d array in which mpg is the 0 element in each row
          return guess - this.labels[i][0];
        })
      ) *
        2) /
      this.features.length;

    // 2/n ∑ -x(Actual - (mx+b))
    // n = number of features = this.features.length
    const mSlope =
      (_.sum(
        currentGuessesForMPG.map((guess, i) => {
          // -1 * this.features[i][0] is equivalent to -x in the above equation
          return -1 * this.features[i][0] * (this.labels[i][0] - guess);
        })
      ) *
        2) /
      this.features.length;

    this.m = this.m - mSlope * this.options.learningRate;
    this.b = this.b - bSlope * this.options.learningRate;
  }

  train() {
    for (let i = 0; i < this.options.iterations; i++) {
      this.gradientDescent();
    }
  }
}

module.exports = LinearRegression;
