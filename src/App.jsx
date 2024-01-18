import { useEffect, useRef, useState } from "react";
import "./App.css";
import * as tf from "@tensorflow/tfjs";
import keypointLabels from "./assets/keypoint_labels.json";
import { renderBoxes, renderPoints } from "./utils/canvas.js";

const numClass = keypointLabels.length;

function App() {
  const [loading, setLoading] = useState({ loading: true, progress: 0 });
  const [model, setModel] = useState({
    net: null,
    inputShape: [1, 0, 0, 3],
  });

  const imageRef = useRef();
  const canvasRef = useRef();

  const preprocess = (source, modelWidth, modelHeight) => {
    let xRatio, yRatio; // ratios for boxes

    const input = tf.tidy(() => {
      const img = tf.browser.fromPixels(source);

      const [h, w] = img.shape.slice(0, 2);
      const maxSize = Math.max(w, h); // get max size
      const imgPadded = img.pad([
        [0, maxSize - h],
        [0, maxSize - w],
        [0, 0],
      ]);

      xRatio = maxSize / w;
      yRatio = maxSize / h;

      return tf.image
        .resizeBilinear(imgPadded, [modelWidth, modelHeight])
        .div(255.0)
        .expandDims(0);
    });

    return [input, xRatio, yRatio];
  };

  const detect = async (source, model, canvasRef, callback = () => {}) => {
    const [modelWidth, modelHeight] = model.inputShape.slice(1, 3);

    tf.engine().startScope();
    const [input, xRatio, yRatio] = preprocess(source, modelWidth, modelHeight);

    const res = model.net.execute(input);
    const transRes = res.transpose([0, 2, 1]);
    const boxes = tf.tidy(() => {
      const w = transRes.slice([0, 0, 2], [-1, -1, 1]);
      const h = transRes.slice([0, 0, 3], [-1, -1, 1]);
      const x1 = tf.sub(transRes.slice([0, 0, 0], [-1, -1, 1]), tf.div(w, 2));
      const y1 = tf.sub(transRes.slice([0, 0, 1], [-1, -1, 1]), tf.div(h, 2));

      return tf.concat([y1, x1, tf.add(y1, h), tf.add(x1, w)], 2).squeeze();
    });

    const [scores, classes] = tf.tidy(() => {
      const rawScores = transRes
        .slice([0, 0, 4], [-1, -1, numClass])
        .squeeze(0);
      return [rawScores.max(1), rawScores.argMax(1)];
    });

    const landmarks = tf.tidy(() => {
      return transRes.slice([0, 0, 5], [-1, -1, -1]).squeeze();
    });

    const nms = await tf.image.nonMaxSuppressionAsync(
      boxes,
      scores,
      1,
      0.45,
      0.3,
    );

    const boxes_data = boxes.gather(nms, 0).dataSync();
    const scores_data = scores.gather(nms, 0).dataSync();
    const classes_data = classes.gather(nms, 0).dataSync();
    let landmarks_data = landmarks.gather(nms, 0).dataSync();

    renderBoxes(
      canvasRef,
      keypointLabels,
      boxes_data,
      scores_data,
      classes_data,
      [xRatio, yRatio],
    );

    renderPoints(canvasRef, landmarks_data, [xRatio, yRatio]);

    tf.dispose([res, transRes, boxes, scores, classes, nms]);

    callback();

    tf.engine().endScope();
  };

  const loadModel = async () => {
    await tf.ready();

    const yolov8 = await tf.loadGraphModel(
      `${window.location.href}/models/keypoint_detection/model.json`,
      {
        onProgress: (fractions) => {
          setLoading({ loading: true, progress: fractions });
        },
      },
    );

    setLoading({ loading: false, progress: 1 });

    setModel({
      net: yolov8,
      inputShape: yolov8.inputs[0].shape,
    });
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    const reader = new FileReader();

    reader.onload = (e) => {
      imageRef.current.src = e.target.result;
    };

    reader.readAsDataURL(file);
  };

  useEffect(() => {
    loadModel();
  }, []);

  if (loading.loading) {
    return (
      <div>
        <h1>Loading model...</h1>
        <p>{(loading.progress * 100).toFixed(2)}%</p>
      </div>
    );
  }

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
      }}
    >
      <h1>YOLOv8 Model With TensorFlow.js</h1>
      <input
        onChange={handleFileChange}
        type="file"
        style={{
          marginBottom: "2rem",
        }}
      />
      <div
        style={{
          position: "relative",
          width: model.inputShape[1],
          height: model.inputShape[2],
        }}
      >
        <img
          alt=""
          ref={imageRef}
          width={model.inputShape[1]}
          height={model.inputShape[2]}
        />
        <canvas
          style={{ position: "absolute", left: 0, top: 0 }}
          width={model.inputShape[1]}
          height={model.inputShape[2]}
          ref={canvasRef}
        />
        <button
          style={{
            marginTop: "1rem",
          }}
          onClick={() => {
            detect(imageRef.current, model, canvasRef.current, () => {
              console.log("done");
            });
          }}
        >
          Run Model
        </button>
      </div>
    </div>
  );
}

export default App;
