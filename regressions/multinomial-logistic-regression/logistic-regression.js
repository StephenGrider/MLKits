const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class LogisticRegression {
  constructor(features, labels, options) {
    this.features = this.processFeatures(features);

    // Labels Tensor
    // assuming labels have been encoded.
    this.labels = tf.tensor(labels);

    // MSE or Cross Entropy. In this case, for logistical regression it's Cross Entropy
    this.costHistory = [];

    this.options = Object.assign(
      {
        learningRate: 0.1,
        iterations: 1000,
        decisionBoundary: 0.5,
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
    this.weights = tf.zeros([this.features.shape[1], this.labels.shape[1]]);
  }

  gradientDescent(features, labels) {
    const currentGuesses = features.matMul(this.weights).softmax();

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
      this.recordCost();
      this.updateLearningRate();
    }
  }

  predict(observations) {
    // prettier-ignore
    return this.processFeatures(observations)
      .matMul(this.weights)
      .softmax()
      .argMax(1); // get max value across columns
  }

  test(testFeatures, testLabels) {
    /**
     * Since our decision boundary is .5, we can use the round() function to
     * make our probability into predictions. Remember this.predict(features)
     * will return the probability between 0 and 1.
     */
    const predictions = this.predict(testFeatures);
    testLabels = tf.tensor(testLabels).argMax(1);

    // prettier-ignore
    const incorrect = predictions
                      .notEqual(testLabels)
                      .sum()
                      .get();

    return (predictions.shape[0] - incorrect) / predictions.shape[0];
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

  recordCost() {
    /**
     * calculating Cross Entropy
     * - 1/n ∑ actual * log(guess) + (1 - actual) * log(1 - guess)
     *        |----- term 1 -----| + |------- term 2 ------------|
     *                               (- actual + 1) * log(- guess + 1)
     */
    const guesses = this.features.matMul(this.weights).softmax();
    // prettier-ignore
    const termOne = this.labels
      .transpose()
      .matMul(guesses.log());

    // prettier-ignore
    const termTwo = this.labels
      .mul(-1)
      .add(1)
      .transpose()
      .matMul(
        guesses.mul(-1)
          .add(1)
          .log()
      );

    const cost = termOne.add(termTwo).div(this.features.shape[0]).mul(-1).get(0, 0);
    this.costHistory.unshift(cost);
  }

  updateLearningRate() {
    if (this.costHistory.length < 2) {
      return;
    }

    // if our guesses are getting worse, then decrease learning rate
    if (this.costHistory[0] > this.costHistory[1]) {
      this.options.learningRate /= 2;
    } else {
      // increase learning rate since our guess was an improvement
      this.options.learningRate *= 1.05;
    }
  }
}

module.exports = LogisticRegression;
