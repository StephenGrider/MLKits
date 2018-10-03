//////////////////////////////////////////
// Hi!
// You probably don't need to edit me
//////////////////////////////////////////

const BALL_SIZE = 16;
const CANVAS_HEIGHT = 600;
const CANVAS_WIDTH = 794;
const PEG_X = 70;
const PEG_Y = 70;
const BUCKET_COLOR = '#b2ebf2';
const COLORS = [
  '#e1f5fe',
  '#b3e5fc',
  '#81d4fa',
  '#4fc3f7',
  '#29b6f6',
  '#03a9f4',
  '#039be5',
  '#0288d1',
  '#0277bd',
  '#01579b'
];

const Engine = Matter.Engine,
  Render = Matter.Render,
  World = Matter.World,
  Bodies = Matter.Bodies,
  Events = Matter.Events,
  Body = Matter.Body;

const engine = Engine.create({
  timing: { timeScale: 2 }
});
const render = Render.create({
  element: document.querySelector('.target'),
  engine: engine,
  options: {
    width: CANVAS_WIDTH,
    height: CANVAS_HEIGHT,
    wireframes: false,
    background: '#f1f1f1'
  }
});

const ground = Bodies.rectangle(
  CANVAS_WIDTH / 2,
  CANVAS_HEIGHT,
  CANVAS_WIDTH * 3,
  50,
  {
    id: 999,
    isStatic: true,
    collisionFilter: { group: 'ground' }
  }
);
const ground2 = Bodies.rectangle(0, CANVAS_HEIGHT, CANVAS_WIDTH * 3, 50, {
  id: 9999,
  isStatic: true,
  collisionFilter: { group: 'ground' }
});
const indicator = Bodies.circle(BALL_SIZE, BALL_SIZE, BALL_SIZE, {
  isStatic: true,
  collisionFilter: { group: 'ball' }
});

const pegs = [];
for (let i = 1; i < CANVAS_HEIGHT / PEG_Y - 1; i++) {
  for (let j = 1; j < CANVAS_WIDTH / PEG_X + 1; j++) {
    let x = j * PEG_X - BALL_SIZE * 1.5;
    const y = i * PEG_Y;

    if (i % 2 == 0) {
      x -= PEG_X / 2;
    }

    const peg = Bodies.polygon(x, y, 7, BALL_SIZE / 4, {
      isStatic: true
    });
    pegs.push(peg);
  }
}

const leftWall = Bodies.rectangle(
  -1,
  CANVAS_HEIGHT / 2 + BALL_SIZE * 2,
  1,
  CANVAS_HEIGHT,
  {
    isStatic: true
  }
);
const rightWall = Bodies.rectangle(
  CANVAS_WIDTH + 1,
  CANVAS_HEIGHT / 2 + BALL_SIZE * 2,
  1,
  CANVAS_HEIGHT,
  {
    isStatic: true
  }
);

const buckets = [];
const bucketIdRange = [];
const bucketWidth = CANVAS_WIDTH / 10;
const bucketHeight = BALL_SIZE * 3;
for (let i = 0; i < 10; i++) {
  const bucket = Bodies.rectangle(
    bucketWidth * i + bucketWidth * 0.5,
    CANVAS_HEIGHT - bucketHeight,
    bucketWidth,
    bucketHeight,
    {
      id: i,
      isStatic: true,
      isSensor: true,
      render: {
        fillStyle: BUCKET_COLOR
      },
      collisionFilter: {
        group: 'bucket'
      }
    }
  );
  const divider = Bodies.rectangle(
    bucketWidth * i,
    CANVAS_HEIGHT - bucketHeight,
    2,
    bucketHeight,
    {
      isStatic: true,
      collisionFilter: { group: 'bucket' }
    }
  );
  bucketIdRange.push(i);
  buckets.push(bucket);
  buckets.push(divider);
}

World.add(engine.world, [
  ground2,
  ...pegs,
  ...buckets,
  ground,
  indicator,
  leftWall,
  rightWall
]);
Engine.run(engine);
Render.run(render);
let ballCount = 0;
function dropBalls(position, quantity) {
  const balls = [];

  const startRes = Math.min(
    Math.abs(parseFloat(document.querySelector('#coef-start').value)),
    1
  );
  const endRes = Math.min(
    Math.abs(parseFloat(document.querySelector('#coef-end').value)),
    1
  );

  const startSize = parseFloat(document.querySelector('#size-start').value);
  const endSize = parseFloat(document.querySelector('#size-end').value);
  for (let i = 0; i < quantity; i++) {
    ballCount++;
    if (ballCount > 785) {
      ballCount--;
      break;
    }
    const restitution = Math.random() * (endRes - startRes) + startRes;
    const size = Math.random() * (endSize - startSize) + startSize;
    const dropX = position;

    const ball = Bodies.circle(dropX, size, size, {
      restitution,
      collisionFilter: { group: 'ball' },
      friction: 0.9
    });
    ball.size = size;
    ball.restitution = restitution;
    ball.dropX = position;
    balls.push(ball);
  }

  World.add(engine.world, balls);
}

let x = 0;
const canvas = document.querySelector('canvas');
const events = {
  mousemove(event) {
    x = event.offsetX;

    Body.setPosition(indicator, { x: x, y: BALL_SIZE });
    document.querySelector('.x-position').innerHTML = `Drop Position: ${x}`;
  },
  click() {
    const quantity = parseInt(document.querySelector('#drop-quantity').value);

    dropBalls(x, quantity);
  }
};
for (let event in events) {
  canvas.addEventListener(event, events[event]);
}

let _score = {};
Events.on(engine, 'collisionActive', ({ pairs }) => {
  const filteredPairs = pairs.forEach(pair => {
    if (
      (bucketIdRange.includes(pair.bodyA.id) ||
        bucketIdRange.includes(pair.bodyB.id)) &&
      Math.abs(pair.bodyB.velocity.y) < 0.1 &&
      pair.bodyB.position.y > CANVAS_HEIGHT - 200
    ) {
      World.remove(engine.world, pair.bodyB);
      ballCount--;
      const bucketId = pair.bodyA.id;

      _score[bucketId] = (_score[bucketId] || 0) + 1;

      const count = parseInt(
        document.querySelector(`#bucket-${bucketId}`).innerHTML
      );
      document.querySelector(`#bucket-${bucketId}`).innerHTML = count + 1;

      onScoreUpdate(
        Math.round(pair.bodyB.dropX),
        pair.bodyB.restitution,
        pair.bodyB.size,
        bucketId + 1
      );
      updateBucketColors(_score);
    }
  });
});

// document.querySelector('button#export').addEventListener('click', () => {
//   const rows = outputs.join('\n');
//   const a = document.createElement('a');
//   mimeType = 'application/octet-stream';
//   a.href = URL.createObjectURL(
//     new Blob([rows], {
//       type: mimeType
//     })
//   );
//   a.setAttribute('download', 'data.csv');
//   document.body.appendChild(a);
//   a.click();
//   document.body.removeChild(a);
// });

document.querySelector('button#scan').addEventListener('click', () => {
  const quantity = parseInt(document.querySelector('#scan-quantity').value);
  const spacing = parseInt(document.querySelector('#scan-spacing').value);

  for (let i = 1; i < CANVAS_WIDTH / spacing; i++) {
    dropBalls(i * spacing, quantity);
  }
});

document.querySelector('button#spot').addEventListener('click', () => {
  const quantity = parseInt(document.querySelector('#spot-quantity').value);
  const spot = parseInt(document.querySelector('#spot-location').value);

  dropBalls(spot, quantity);
});

document.querySelector('button#analyze').addEventListener('click', runAnalysis);

document
  .querySelectorAll('form')
  .forEach(f => f.addEventListener('submit', e => e.preventDefault()));

function updateBucketColors(_score) {
  const counts = _.range(0, 10).map(i => _score[i] || 0);

  const min = _.min(counts);
  const max = _.max(counts);

  const ranks = counts.map((count, i) => ({ i, c: count }));

  let counter = 0;
  const d = _.chain(ranks)
    .sortBy('c')
    .forEach(({ i, c }, j, collection) => {
      if (_.get(collection, `[${j - 1}].c`) !== c) {
        counter++;
      }
      buckets[i * 2].render.fillStyle = COLORS[counter - 1];
    })
    .value();
}

document.querySelector('#reset').addEventListener('click', function() {
  try {
    while (outputs.length) {
      outputs.pop();
    }
  } catch (e) {}

  _.range(0, 10).forEach(i => {
    buckets[i * 2].render.fillStyle = BUCKET_COLOR;
    document.querySelector(`#bucket-${i}`).innerHTML = 0;
  });

  _score = {};
});
