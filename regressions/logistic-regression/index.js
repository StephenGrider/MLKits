require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const plot = require('node-remote-plot');
const loadCSV = require('../load-csv');
const LogisticRegression = require('./logistic-regression');

const { features, labels, testFeatures, testLabels } = loadCSV('../data/cars.csv', {
  dataColumns: ['horsepower', 'displacement', 'weight'],
  labelColumns: ['passedemissions'],
  shuffle: true,
  splitTest: 50,
  converters: {
    passedemissions: (value) => {
      return value === 'TRUE' ? 1 : 0;
    },
  },
});

const regress = new LogisticRegression(features, labels, {
  learningRate: 0.5,
  iterations: 100,
  batchSize: 10,
  decisionBoundary: 0.5,
});

regress.train();

console.log('% correct: ', regress.test(testFeatures, testLabels));
plot({
  x: regress.costHistory.reverse(),
  xLabel: 'iterations',
  yLabel: 'cost',
});
