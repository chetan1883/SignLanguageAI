import React, { useEffect, useRef, useState } from "react";

import Webcam from "react-webcam";
import axios from "axios";

import { Hands } from "@mediapipe/hands";
import { Camera } from "@mediapipe/camera_utils";

import {
  drawConnectors,
  drawLandmarks,
} from "@mediapipe/drawing_utils";

import { HAND_CONNECTIONS } from "@mediapipe/hands";

function App() {

  const webcamRef = useRef(null);
  const canvasRef = useRef(null);

  const [prediction, setPrediction] = useState("");
  const [confidence, setConfidence] = useState(0);
  const [word, setWord] = useState("");

  const lastPredictionRef = useRef("");
  const lastAddedTimeRef = useRef(Date.now());

  const predictionBufferRef = useRef([]);

  useEffect(() => {

    const hands = new Hands({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
      },
    });

    hands.setOptions({
      maxNumHands: 1,
      modelComplexity: 1,
      minDetectionConfidence: 0.7,
      minTrackingConfidence: 0.7,
    });

    hands.onResults(onResults);

    if (
      typeof webcamRef.current !== "undefined" &&
      webcamRef.current !== null
    ) {

      const camera = new Camera(
        webcamRef.current.video,
        {
          onFrame: async () => {

            await hands.send({
              image: webcamRef.current.video,
            });

          },

          width: 640,
          height: 480,
        }
      );

      camera.start();
    }

  }, []);

  const onResults = async (results) => {

    const canvasElement = canvasRef.current;

    const canvasCtx =
      canvasElement.getContext("2d");

    canvasCtx.save();

    canvasCtx.clearRect(
      0,
      0,
      canvasElement.width,
      canvasElement.height
    );

    canvasCtx.drawImage(
      results.image,
      0,
      0,
      canvasElement.width,
      canvasElement.height
    );

    if (results.multiHandLandmarks) {

      for (const landmarks of results.multiHandLandmarks) {

        drawConnectors(
          canvasCtx,
          landmarks,
          HAND_CONNECTIONS
        );

        drawLandmarks(
          canvasCtx,
          landmarks
        );

        let data = [];

        landmarks.forEach((lm) => {

          data.push(lm.x);
          data.push(lm.y);

        });

        try {

          const response = await axios.post(
            " https://signlanguageai-l0zo.onrender.com/predict",
            {
              landmarks: data,
            }
          );

          const pred =
            response.data.prediction;

          const conf =
            response.data.confidence;

          setConfidence(conf);

          predictionBufferRef.current.push(pred);

          if (
            predictionBufferRef.current.length > 10
          ) {
            predictionBufferRef.current.shift();
          }

          const counts = {};

          predictionBufferRef.current.forEach((p) => {

            counts[p] =
              (counts[p] || 0) + 1;

          });

          const stablePrediction =
            Object.keys(counts).reduce((a, b) =>
              counts[a] > counts[b]
                ? a
                : b
            );

          setPrediction(stablePrediction);

          if (
            stablePrediction !==
            lastPredictionRef.current
          ) {

            lastPredictionRef.current =
              stablePrediction;

            window.speechSynthesis.cancel();

            const speech =
              new SpeechSynthesisUtterance(
                stablePrediction
              );

            window.speechSynthesis.speak(
              speech
            );
          }

          const now = Date.now();

          if (
            now - lastAddedTimeRef.current > 2000
          ) {

            setWord((prev) =>
              prev + stablePrediction
            );

            lastAddedTimeRef.current = now;
          }

        } catch (error) {

          console.log(error);

        }
      }
    }

    canvasCtx.restore();
  };

  const speakWord = () => {

    window.speechSynthesis.cancel();

    const speech =
      new SpeechSynthesisUtterance(word);

    window.speechSynthesis.speak(speech);
  };

  const clearWord = () => {
    setWord("");
  };

  const addSpace = () => {
    setWord((prev) => prev + " ");
  };

  const removeLast = () => {
    setWord((prev) => prev.slice(0, -1));
  };

  return (

    <div className="min-h-screen bg-slate-900 text-white p-6">

      <h1 className="text-5xl font-bold text-center mb-8">
        SignLanguageAI
      </h1>

      <div className="flex flex-col lg:flex-row gap-8 justify-center items-center">

        <div className="relative w-[640px] h-[480px] rounded-3xl overflow-hidden border-4 border-cyan-400 shadow-2xl">

          <Webcam
            ref={webcamRef}
            className="absolute w-full h-full object-cover"
          />

          <canvas
            ref={canvasRef}
            width={640}
            height={480}
            className="absolute w-full h-full"
          />

        </div>

        <div className="bg-slate-800 p-8 rounded-3xl shadow-2xl w-[350px]">

          <h2 className="text-3xl font-bold mb-4">
            Live Prediction
          </h2>

          <div className="mb-6">

            <p className="text-xl text-cyan-400">
              Letter
            </p>

            <h1 className="text-7xl font-bold">
              {prediction}
            </h1>

          </div>

          <div className="mb-6">

            <p className="text-xl text-cyan-400">
              Confidence
            </p>

            <h2 className="text-3xl font-bold">
              {confidence}%
            </h2>

          </div>

          <div className="mb-6">

            <p className="text-xl text-cyan-400">
              Word
            </p>

            <h2 className="text-3xl font-bold break-words">
              {word}
            </h2>

          </div>

          <div className="grid grid-cols-2 gap-4">

            <button
              onClick={speakWord}
              className="bg-cyan-500 hover:bg-cyan-600 p-3 rounded-xl font-bold"
            >
              Speak
            </button>

            <button
              onClick={clearWord}
              className="bg-red-500 hover:bg-red-600 p-3 rounded-xl font-bold"
            >
              Clear
            </button>

            <button
              onClick={addSpace}
              className="bg-green-500 hover:bg-green-600 p-3 rounded-xl font-bold"
            >
              Space
            </button>

            <button
              onClick={removeLast}
              className="bg-yellow-500 hover:bg-yellow-600 p-3 rounded-xl font-bold text-black"
            >
              Backspace
            </button>

          </div>

        </div>

      </div>

    </div>
  );
}

export default App;