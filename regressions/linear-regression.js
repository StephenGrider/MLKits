const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class LinearRegression {
  constructor(features, labels, options) {
    this.features = this.processFeatures(features);

    // Labels Tensor
    this.labels = tf.tensor(labels);

    this.mseHistory = [];

    this.options = Object.assign(
      {
        learningRate: 0.1,
        iterations: 1000,
      },
      options
    );

    /**
     * weights tensor
     * by convention initial guesses are given the value of either 0 or 1
     *
     * resulting tensor has a [n,1] shape and looks like this, where n is the
     * number of colums present in the features tensor after adding the
     * column of 1s :
     * [
     *  [0],
     *  [0],
     * ...
     * ]
     *
     */
    this.weights = tf.zeros([this.features.shape[1], 1]);
  }

  gradientDescent() {
    /**
     * Matrix Multiplication
     *
     * Slope of MSE       features * ((features * weights) - labels)
     * with respect  =    ------------------------------------------
     * to m & b                             n
     *
     * n = total number of features.
     * (features * weights)
     *    -> is essentially the (mx + b) portion of the MSE formula
     *    -> here is represented as `currentGuesses` below
     *
     * (currentGuesses - labels) aka ((features * weights) - labels)
     *    -> is represented below as `differences`
     *
     * (features * differences) / n
     *    -> is represented below as `slopes`
     *    -> `features` is first transposed prior to multiplying by `differences`
     */

    /**
     * features shape: [n, c] where n is number of records (rows) and
     *    `c` is the number of columns or unique
     *    features (e.g., horsepoer, weight, etc, displacement)
     *
     * weights shape: [2,1]
     *
     * initially we're only using one feature so the features size is [n, 1], but
     * we prepended a column of 1s to the features tensor, so it now has the
     * shape of [n,2] so that we could multiply them, e.g., [n,2][2,1] works.
     *
     * currentGuesses will then have the shape of [n,1]
     */
    const currentGuesses = this.features.matMul(this.weights);

    /**
     * elementwise subtraction of the `labels` from `currentGuesses`.
     * Will this be problematic if we have more than 1 label? In other words,
     * shouldn't this be matrix subtraction?
     *
     * Elementwise subtraction would render `differences` with the same shape
     * that currentGuesses had of [n,1]
     */
    const differences = currentGuesses.sub(this.labels);

    /**
     * transpose features from an [n, c] to a [c, n] shape.
     *  -> initially `c` = 2 (feature, plus column of 1s)
     *  -> n is the number of records/rows.
     *
     * `differences` has an [n,1] shape so multiplying [c, n][n, 1] shapes works.
     *
     * Remember that tensors are immutable, so when we then divide by
     * the number of records, we can use features.shape[0] because this.features
     * is still the shape [n, 2]
     */
    const slopes = this.features.transpose().matMul(differences).div(this.features.shape[0]);

    // update the weights by multiplying the just calculated slopes by the learning rate
    this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
  }

  train() {
    for (let i = 0; i < this.options.iterations; i++) {
      this.gradientDescent();
      this.recordMSE();
      this.updateLearningRate();
    }
  }

  test(testFeatures, testLabels) {
    // convert multidimensinal arrays to tensors
    testFeatures = this.processFeatures(testFeatures);
    testLabels = tf.tensor(testLabels);
    const predictions = testFeatures.matMul(this.weights);

    /**
     * Sum of Squares Residual
     * ------------------------
     * ∑ (actual - predicted) ^2
     *
     * 1. testLabels contains all of our actual mpg values
     * 2. we subtract our `predictions` based on feature horsepower
     * 3. we square that difference
     * 4. add all those squares together into a single [1,1] tensor. Note that
     *    we don't use an axis argument in the sum() method because we want
     *    all the values in the matrix added together into a single value as
     *    opposed to adding all the columns together or all the rows.
     * 5. get the value of that summation
     *
     */

    // prettier-ignore
    const ss_res = testLabels           // 1. 
                    .sub(predictions)   // 2. 
                    .pow(2)             // 3.
                    .sum()              // 4.
                    .get(); // 5.

    /**
     * Sum of Squares Total
     * ------------------------
     * ∑ (actual - average) ^2
     *
     * We'll follow the process outlined above for ss_res, substituting the
     * `average` value for mpg rather than the predicted value.
     */

    const ss_tot = testLabels.sub(testLabels.mean()).pow(2).sum().get();

    // return the coefficient of determination R^2
    // R^2 = 1 - (ss_res / ss_tot)
    return 1 - ss_res / ss_tot;
  }

  /**
   * processFeatures
   * @param {array} features
   *
   * 1. cast features Array into a tensor
   * 2. prepend a column of 1s
   * 3. return new tensor.
   */
  processFeatures(features) {
    /**
     * Cast Array into Features Tensor
     *
     * Initially, features is passed in as a JS array and just has a single
     * feature (horsepower) and looks something like this with an [n, 1] shape.
     *
     * [
     *  [88],
     *  [152],
     *  [245],
     *  ...
     * ]
     *
     */
    features = tf.tensor(features);

    features = this.standardize(features);
    /**
     * prepend a column of `1s` to the features tensor so that it now looks
     * something like this with a [n, 2] shape:
     * [
     *  [1, n],
     *  [1, n],
     *  [1, n],
     *  ...
     * ]
     */
    features = tf.ones([features.shape[0], 1]).concat(features, 1);

    return features;
  }

  /**
   * Standarize
   */
  standardize(features) {
    const { mean, variance } = tf.moments(features, 0);

    // if instance variables are not defined, define them, otherwise, use
    // the previously defined value.
    this.mean = this.mean || mean;
    this.variance = this.variance || variance;

    return features.sub(this.mean).div(this.variance.pow(0.5));
  }

  recordMSE() {
    /**
     * calculating MSE
     * 1/n ∑ ((features * weights) - labels)^2)
     */
    // prettier-ignore
    const mse = this.features
                  .matMul(this.weights)
                  .sub(this.labels)
                  .pow(2)
                  .sum()
                  .div(this.features.shape[0])
                  .get();

    // place current mse at the top of the array
    this.mseHistory.unshift(mse);
  }

  updateLearningRate() {
    if (this.mseHistory.length < 2) {
      return;
    }

    // if our guesses are getting worse, then decrease learning rate
    if (this.mseHistory[0] > this.mseHistory[1]) {
      this.options.learningRate /= 2;
    } else {
      // increase learning rate since our guess was an improvement
      this.options.learningRate *= 1.05;
    }
  }
}

module.exports = LinearRegression;
