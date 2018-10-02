const outputs = [];

function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {
  outputs.push([dropPosition, bounciness, bucketLabel]);
}

function runAnalysis() {
  const testSet = getRandomSamples(outputs, 100);

  let dropPositionCorrect = 0;
  let bouncinessCorrect = 0;
  for (let i = 0; i < testSet.length; i++) {
    const dropResult = knn(outputs, testSet[i], 4, 0, 2);
    const bounceResult = knn(outputs, testSet[i], 4, 1, 2);

    if (dropResult === _.last(testSet[i])) {
      dropPositionCorrect++;
    }

    if (bounceResult === _.last(testSet[i])) {
      bouncinessCorrect++;
    }
  }

  console.log(`
    Out of 100 samples 
    dropPosition was correct ${dropPositionCorrect} times, 
    bounciness was correct ${bouncinessCorrect} times.`);
}

function knn(data, guessBucketFor, k, dataIndex, labelIndex) {
  const x = data.map(d => d[dataIndex]);
  const y = data.map(d => d[labelIndex]);

  const distances = _.map(x, (point, i) => [
    Math.abs(point - guessBucketFor[dataIndex]),
    y[i]
  ]);
  const sortedDistances = _.sortBy(distances, pair => pair[0]);
  const groups = _.countBy(_.slice(sortedDistances, 0, k), pair => pair[1]);

  return parseInt(_.maxBy(_.keys(groups), key => groups[key]));
}

// function knn(data, guessBucketFor, k) {
//   // minMax(data);

//   const x = data.map(d => d[0]);
//   const y = data.map(d => d[2]);

//   const distances = _.map(x, (point, i) => [
//     distance(point, guessBucketFor),
//     y[i]
//   ]);
//   const sortedDistances = _.sortBy(distances, pair => pair[0]);
//   const groups = _.countBy(_.slice(sortedDistances, 0, k), pair => pair[1]);

//   return _.maxBy(_.keys(groups), key => groups[key]);
// }

function getRandomSamples(data, numberSamples) {
  let samples = [];

  for (let i = 0; i < numberSamples; i++) {
    const randomIndex = Math.floor(Math.random() * data.length);

    samples.push(...data.splice(randomIndex, 1));
  }

  return samples;
}

function distance(pointA, pointB) {
  const squaredDifferences = pointB.map((val, i) => {
    return (pointB[i] - pointA[i]) ** 2;
  });

  return _.sum(squaredDifferences) ** 0.5;
}

function minMax(data) {
  for (let i = 0; i < data[0].length - 1; i++) {
    const column = data.map(d => d[i]);

    const min = _.min(column);
    const max = _.max(column);

    for (let j = 0; j < data.length; j++) {
      data[j][i] = (data[j][i] - min) / (max - min);
    }
  }
}

function findIdealK() {
  const testSet = getRandomSamples(outputs, 100);

  const report = {};
  for (let k = 1; k < 5; k++) {
    report[k] = 0;

    for (let i = 0; i < testSet.length; i++) {
      const guessedBucket = knn(outputs, testSet[i], k);

      if (guessedBucket === _.last(testSet[i])) {
        report[k] += 1;
      }
    }
  }

  console.log(report);
}
