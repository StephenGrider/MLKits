const outputs = [];
// determine the corellation between bucket and predictionPoint

function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {
  outputs.push([dropPosition, bounciness, size, bucketLabel]);
}

function runAnalysis() {
  const testSetSize = 100;
  const k = 10;
  const colNames = ['Drop Position', 'Bounciness', 'Ball Size'];

  // vary k using range
  _.range(0, 3).forEach((feature) => {
    const data = _.map(outputs, (row) => [row[feature], _.last(row)]);

    const [testSet, trainingSet] = splitDataSet(minMax(data, 1), testSetSize);
    const accuracy = _.chain(testSet)
      .filter((testPoint) => knn(trainingSet, _.initial(testPoint), k) === _.last(testPoint))
      .size()
      .divide(testSetSize)
      .value();

    console.log(`k(${k}) Accuracy for ${colNames[feature]}: ${accuracy * 100}%`);
  });
}

function knn(data, point, k) {
  // K-Nearest Neighbor Algorithm
  return (
    _.chain(data)
      // [[distance(dropPosition, predictionPoint), bucketLabel],[72,4],[227,5]]
      .map((row) => {
        return [distance(_.initial(row), point), _.last(row)];
      })
      // sort by drop position
      .sortBy((row) => row[0])
      // Gets the top 'k' results from sorted list
      .slice(0, k)
      // counts frequency of buckets
      // e.g.,  {"3":1,"4":2}
      .countBy((row) => row[1])
      // e.g., [["3",1],["4",2]]
      // converts the countBy obj to an multidimensional array
      .toPairs()
      // sorts so that the most frequent is the last array element
      .sortBy((row) => row[1])
      // get the last array element of ["bucket", frequency]
      .last()
      // e.g., "4"
      // get the bucket number (first element)
      .first()
      // e.g., 4
      // convert the string "4" to int 4
      .parseInt()
      // end the chain and return the value
      .value()
  );
}

function distance(pointA, pointB) {
  // pointA/B are arrays
  // employing the pythagorean therom to solve a multidimensional point distance
  _.chain(pointA)
    // takes each value at the same index of each array and creates a new zipped
    // array:
    // [[pointA[0], pointB[0]], [pointA[1], pointB[1]] ... ]
    .zip(pointB)
    // subtracts b from a
    .map(([a, b]) => (a - b) ** 2)
    // sums the squares
    .sum()
    // returns the squareroot of the sum
    .value() ** 0.5;
}

function splitDataSet(data, testCount) {
  const shuffled = _.shuffle(data);

  const testSet = _.slice(shuffled, 0, testCount);
  const trainingSet = _.slice(shuffled, testCount);

  return [testSet, trainingSet];
}

function minMax(data, featureCount) {
  const clonedData = _.cloneDeep(data);

  // iterate over each feature (independent variables)
  for (let i = 0; i < featureCount; i++) {
    const column = clonedData.map((row) => row[i]);
    const min = _.min(column);
    const max = _.max(column);

    // iterate over each row [j] in clonedData
    // and normalize each feature [i]
    // a row would look something like :
    //
    // [ position, bounciness, ballSize, bucketName ]
    //
    // where bucketName is a label and not normalized
    for (let j = 0; j < clonedData.length; j++) {
      clonedData[j][i] = (clonedData[j][i] - min) / (max - min);
      if (max - min === 0) {
        clonedData[j][i] = 0;
      }
    }
  }

  return clonedData;
}
