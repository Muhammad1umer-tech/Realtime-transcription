// client.js
let socket;
let audioContext;
let processor;
let globalStream;

const startButton = document.getElementById('startButton');
const stopButton = document.getElementById('stopButton');
const transcriptionDiv = document.getElementById('transcription');

startButton.addEventListener('click', startTranscription);
stopButton.addEventListener('click', stopTranscription);

function startTranscription() {
  startButton.disabled = true;
  stopButton.disabled = false;

  // Initialize WebSocket connection to the server
  socket = new WebSocket('ws://192.168.2.30:8900/ws/audio');

  socket.onopen = function() {
    console.log('WebSocket connection established.');
  };

  socket.onmessage = function(event) {
    const message = event.data;
    transcriptionDiv.innerHTML += `<p>${message}</p>`;
  };

  socket.onclose = function() {
    console.log('WebSocket connection closed.');
  };

  // Start capturing audio
  navigator.mediaDevices.getUserMedia({ audio: true })
    .then(stream => {
      globalStream = stream;

      audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const input = audioContext.createMediaStreamSource(stream);

      processor = audioContext.createScriptProcessor(4096, 1, 1);
      processor.onaudioprocess = function(e) {
        const inputData = e.inputBuffer.getChannelData(0);
        // Convert Float32Array to Int16Array
        const int16Data = floatTo16BitPCM(inputData);
        // Send audio data to server
        if (socket.readyState === WebSocket.OPEN) {
          socket.send(int16Data);
        }
      };

      input.connect(processor);
      processor.connect(audioContext.destination);
    })
    .catch(err => {
      console.error('Error accessing microphone:', err);
    });
}

function stopTranscription() {
  startButton.disabled = false;
  stopButton.disabled = true;

  // Close WebSocket connection
  if (socket) {
    socket.close();
  }

  // Stop audio processing
  if (processor) {
    processor.disconnect();
  }
  if (audioContext) {
    audioContext.close();
  }
  if (globalStream) {
    globalStream.getTracks().forEach(track => track.stop());
  }
}
function floatTo16BitPCM(float32Array) {
  const int16Array = new Int16Array(float32Array.length);
  for (let i = 0; i < float32Array.length; i++) {
    const s = Math.max(-1, Math.min(1, float32Array[i]));
    int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
  }
  return int16Array;
}

