const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class LinearRegression {
  constructor(features, labels, options) {
    /**
     * Features Tensor
     *
     * Initially just has a single feature (horsepower) and looks something like
     * this with an [n, 1] shape:
     *
     * [
     *  [88],
     *  [152],
     *  [245],
     *  ...
     * ]
     *
     */
    this.features = tf.tensor(features);
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
    this.features = tf.ones([this.features.shape[0], 1]).concat(this.features, 1);

    // Labels Tensor
    this.labels = tf.tensor(labels);

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
     * resulting tensor has a [2,1] shape and looks like this:
     * [
     *  [0],
     *  [0]
     * ]
     *
     */
    this.weights = tf.zeros([2, 1]);
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
    }
  }

  test(testFeatures, testLabels) {
    // convert multidimensinal arrays to tensors
    testFeatures = tf.tensor(testFeatures);
    testLabels = tf.tensor(testLabels);

    // prepend column of 1s to testFeatures
    testFeatures = tf.ones([testFeatures.shape[0], 1]).concat(testFeatures, 1);
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
}

module.exports = LinearRegression;
