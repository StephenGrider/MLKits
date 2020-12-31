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

  gradientDescent(features, labels) {
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
    const currentGuesses = features.matMul(this.weights);

    /**
     * Elementwise subtraction would render `differences` with the same shape
     * that currentGuesses had of [n,1]
     */
    const differences = currentGuesses.sub(labels);

    /**
     * transpose features from an [n, c] to a [c, n] shape.
     *  -> initially `c` = 2 (feature, plus column of 1s)
     *  -> n is the number of records/rows.
     *
     * `differences` has an [n,1] shape so multiplying [c, n][n, 1] shapes works.
     *
     * Remember that tensors are immutable, so when we then divide by
     * the number of records, we can use features.shape[0] because features
     * is still the shape [n, 2]
     */
    const slopes = features.transpose().matMul(differences).div(features.shape[0]);

    // update the weights by multiplying the just calculated slopes by the learning rate
    this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
  }

  train() {
    // determine the number of batches needed to process the dataset
    const batchQuantity = Math.floor(this.features.shape[0] / this.options.batchSize);

    for (let i = 0; i < this.options.iterations; i++) {
      for (let j = 0; j < batchQuantity; j++) {
        /**
         * Example for start index
         * -----------------------
         * If 88 records and batchSize of 10, batchQuantity is 9.
         * So: j * batchSize iterations look like this:
         *    0 * 10 = 0
         *    1 * 10 = 10
         *    2 * 10 = 20
         *    ...
         *    8 * 10 = 80 (only having 8 records)
         *
         *  In this way we accurately have the record number to start with for
         * the next batch slice
         */
        const startIndex = j * this.options.batchSize;

        /**
         * Slicing the features
         * --------------------
         * Slicing a 2D tensor requires a starting coord and a shape of the
         * slice you want to take.
         *
         * tensor.slice([0,0], [10,-1]) would slice tensor from the first
         * row,col and return 10 rows with as many columns as present.
         *
         * tensor.slice([10,0], [10,-1]) would slice tensor from the 10th row
         * and first column returning 10 rows with as many columns as present.
         *
         * By multiplying j * batchSize we always have the correct row index and
         * we always want the first column so [j * batchSize, 0] is perfect. By
         * using [batchSize,-1] we always have the correct shape to extract.
         */
        const featuresSlice = this.features.slice([startIndex, 0], [this.options.batchSize, -1]);
        /**
         * We need the correct number of labels to process so that our Matricies
         * are the right sizes for multiplication in gradientDescent.
         */
        const labelsSlice = this.labels.slice([startIndex, 0], [this.options.batchSize, -1]);
        this.gradientDescent(featuresSlice, labelsSlice);
      }
      // make updates after processing each batch
      this.recordMSE();
      this.updateLearningRate();
    }
  }

  predict(observations) {
    return this.processFeatures(observations).matMul(this.weights);
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
