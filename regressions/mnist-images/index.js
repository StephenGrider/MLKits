require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const plot = require('node-remote-plot');
const LogisticRegression = require('./logistic-regression');
const _ = require('lodash');
const mnist = require('mnist-data');

// --------- Helper Functions -----------------
const encodeLabelValues = (labelValues) => {
  return labelValues.map((label) => {
    /**
     * new Array(10).fill(0);
     * creates an array ten elements in length and makes each element a 0.
     * [0,0,0,0,0,0,...]
     */
    const row = new Array(10).fill(0);

    /**
     * set the row at the label index to 1 indicating the encoded value. In the
     * case that the label value = 3, the resuting array would look like this:
     * [0,0,0,1,0,0,...]
     */
    row[label] = 1;

    return row;
  });
};

function loadData() {
  const mnistData = mnist.training(0, 60000);

  // lodash's flat map removes one level of array nesting.
  const features = mnistData.images.values.map((image) => _.flatMap(image));
  const encodedLabels = encodeLabelValues(mnistData.labels.values);

  return { features, labels: encodedLabels };
}

function loadTestData() {
  const testMnistData = mnist.testing(0, 1000);
  const testFeatures = testMnistData.images.values.map((image) => _.flatMap(image));
  const testEncodedlabels = encodeLabelValues(testMnistData.labels.values);
  return { testFeatures, testLabels: testEncodedlabels };
}
// enclose the initialization so that JS Garbage Collector doesn't retain
// unneeded elements.
function init() {
  const { features, labels } = loadData();

  return new LogisticRegression(features, labels, {
    learningRate: 1,
    iterations: 20,
    batchSize: 100,
  });
}
function getAccuracy() {
  const { testFeatures, testLabels } = loadTestData();
  return regress.test(testFeatures, testLabels);
}

// ----- Applicatin Code -------------
regress = init();
regress.train();
debugger;

const accuracy = getAccuracy();
console.log('Accuracy: ', accuracy);
